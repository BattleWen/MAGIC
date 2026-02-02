# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.separated_trainer.ppo.ray_trainer import RayReMASeparatedTrainer

import os
import ray
import hydra
import pdb


def get_custom_reward_fn(config):
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
    # isolation, will solve in the future
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
                'RAY_DEBUG': 'legacy',
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        switch_cfg = config.algorithm.switch_agent
        agent_roles = list(switch_cfg.agent_roles) if switch_cfg.agent_roles is not None else []
        model_paths = list(switch_cfg.model_paths) if switch_cfg.model_paths is not None else []
        if not agent_roles:
            raise ValueError("algorithm.switch_agent.agent_roles must be specified.")
        if not model_paths:
            base_path = config.actor_rollout_ref.model.path
            if not base_path:
                raise ValueError("Either actor_rollout_ref.model.path or algorithm.switch_agent.model_paths must be set.")
            model_paths = [base_path for _ in agent_roles]
        elif len(model_paths) == 1 and len(agent_roles) > 1:
            model_paths = model_paths * len(agent_roles)
        elif len(model_paths) != len(agent_roles):
            raise ValueError("algorithm.switch_agent.model_paths must align with agent_roles.")

        if not config.actor_rollout_ref.model.path:
            config.actor_rollout_ref.model.path = model_paths[0]

        agent_tokenizers = {}
        agent_processors = {}
        for role, model_path in zip(agent_roles, model_paths):
            local_path = copy_to_local(model_path)
            agent_tokenizers[role] = hf_tokenizer(local_path)
            agent_processors[role] = hf_processor(local_path, use_fast=True)

        default_role = agent_roles[-1]
        tokenizer = agent_tokenizers[default_role]
        processor = agent_processors[default_role]

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.fsdp_rema_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            raise NotImplementedError('Megatron is not implemented yet')
            # from verl.workers.megatron_rema_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.separated_trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.Agent0_ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Agent0_Critic: ray.remote(CriticWorker),
            # Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
            Role.Agent1_ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Agent1_Critic: ray.remote(CriticWorker),
        }

        agent0_pool_id = 'agent0_global_pool'
        agent1_pool_id = 'agent1_global_pool'
        resource_pool_spec = {
            agent0_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            agent1_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.Agent0_ActorRollout: agent0_pool_id,
            Role.Agent0_Critic: agent0_pool_id,
            Role.Agent1_ActorRollout: agent1_pool_id,
            Role.Agent1_Critic: agent1_pool_id,
        }

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            assert not config.algorithm.use_kl_in_reward, 'use_kl_in_reward not supported now'
            role_worker_mapping[Role.Agent0_RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.Agent0_RefPolicy] = agent0_pool_id
            role_worker_mapping[Role.Agent1_RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.Agent1_RefPolicy] = agent1_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "rema")
        # if reward_manager_name == 'naive':
        #     from verl.workers.reward_manager import NaiveRewardManager
        #     reward_manager_cls = NaiveRewardManager
        # elif reward_manager_name == 'prime':
        #     from verl.workers.reward_manager import PrimeRewardManager
        #     reward_manager_cls = PrimeRewardManager
        if reward_manager_name == 'rema':
            from verl.workers.reward_manager import ReMARewardManager
            reward_manager_cls = ReMARewardManager
        elif reward_manager_name == 'game':
            from verl.workers.reward_manager import GameRewardManager
            reward_manager_cls = GameRewardManager
        else:
            raise NotImplementedError(f"Reward manager {reward_manager_name} is not implemented")

        compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayReMASeparatedTrainer(config=config,
                                tokenizer=agent_tokenizers,
                                processor=processor,
                                role_worker_mapping=role_worker_mapping,
                                resource_pool_manager=resource_pool_manager,
                                ray_worker_group_cls=ray_worker_group_cls,
                                reward_fn=reward_fn,
                                val_reward_fn=val_reward_fn)
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()
