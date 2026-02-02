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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import pdb
import copy
import os
import math
from pathlib import Path
import uuid
import jsonlines
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type, Dict, List
from copy import deepcopy
from collections import defaultdict, deque

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.separated_trainer.ppo import core_algos
from verl.separated_trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
# from verl.utils.dataset.rema_dataset import ReMADataset, collate_fn
from verl.utils.dataset.game_dataset import GameDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils import torch_functional as verl_F
from verl.separated_trainer.ppo.multi_agent_rollout import MultiAgentRollout
from verl.utils.model import compute_position_id_with_mask



WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Agent0_Actor = 0
    Agent0_Rollout = 1
    Agent0_ActorRollout = 2
    Agent0_Critic = 3
    Agent0_RefPolicy = 4
    Agent0_RewardModel = 5
    Agent0_ActorRolloutRef = 6
    Agent1_Actor = 7
    Agent1_Rollout = 8
    Agent1_ActorRollout = 9
    Agent1_Critic = 10
    Agent1_RefPolicy = 11
    Agent1_RewardModel = 12
    Agent1_ActorRolloutRef = 13


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        import time
        import logging
        
        timeout = 300  # 300 seconds = 5 minutes
        retry_interval = 10  # seconds
        start_time = time.time()
        
        while True:
            node_available_resources = ray.state.available_resources_per_node()
            node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

            # check total required gpus can be satisfied
            total_available_gpus = sum(node_available_gpus.values())
            total_required_gpus = sum(
                [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
            
            # Check for resource pool satisfaction
            pools_satisfied = True
            error_msgs = []
            
            if total_available_gpus < total_required_gpus:
                pools_satisfied = False
                error_msgs.append(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
            else:
                # check each resource pool can be satisfied, O(#resource_pools * #nodes)
                for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
                    num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
                    for node, available_gpus in node_available_gpus.items():
                        if available_gpus >= num_gpus:
                            node_available_gpus[node] -= num_gpus
                            num_nodes -= 1
                            if num_nodes == 0:
                                break
                    if num_nodes > 0:
                        pools_satisfied = False
                        error_msgs.append(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster")
            
            # If all resources are available, return
            if pools_satisfied:
                return
            
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                # If we've timed out, raise the error with all collected error messages
                raise ValueError(f"Resource allocation timed out after {timeout} seconds. Errors: {'; '.join(error_msgs)}")
            
            # Log waiting message and sleep before retry
            remaining = timeout - elapsed_time
            logging.info(f"Waiting for resources to be available. Retrying in {retry_interval} seconds. Timeout in {remaining:.1f} seconds.")
            logging.info(f"Resource issues: {'; '.join(error_msgs)}")
            time.sleep(retry_interval)


import torch
from verl.utils.torch_functional import masked_mean
from verl.utils.reward_score.game import USE_ANSWER_TAGS_FOR_REWARD_MODEL, compute_score_components, format_reward_func


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        raise NotImplementedError('GAE is not implemented yet')
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_sparse_rewards = torch.zeros_like(data.batch['token_level_rewards'])
        grpo_sparse_rewards[:, -1] = data.batch['turn_level_reward'].sum(-1)
        index = data.non_tensor_batch['uid']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=grpo_sparse_rewards,
                                                                        eos_mask=step_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        # responses = data.batch['responses']
        # response_length = responses.size(-1)
        # attention_mask = data.batch['attention_mask']
        # response_mask = attention_mask[:, -response_length:]
        step_mask = data.batch['step_ids'] != -100
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=step_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        raise NotImplementedError('REMAX is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        raise NotImplementedError('RLOO is not implemented yet')
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

def get_last_index_of_turn(step_ids: torch.Tensor, i_turn: int) -> torch.Tensor:
    mask = step_ids == i_turn
    seq_tensor = torch.arange(step_ids.size(1), device=step_ids.device).expand_as(step_ids)
    last_indices = torch.where(mask, seq_tensor, torch.tensor(-1, device=step_ids.device))
    last_indices, _ = torch.max(last_indices, dim=1)  # shape: [bsz]
    
    return last_indices

def compute_token_level_scores(data: DataProto, dtype=torch.float32)->torch.Tensor:
    max_num_turns = data.meta_info['max_num_turns']
    bsz, seq_len = data.batch['input_ids'].shape
    token_level_scores = torch.zeros((bsz, seq_len), dtype=torch.float32)
    step_ids = data.batch['step_ids']
    turn_level_return = data.batch['turn_level_return']
    for i_turn in range(max_num_turns):
        last_indices = get_last_index_of_turn(step_ids, i_turn)
        valid_mask = last_indices != -1
        if (~valid_mask).all(): break
        batch_indices = torch.arange(bsz)
        token_level_scores[batch_indices[valid_mask], last_indices[valid_mask]] = \
            turn_level_return[:, i_turn][valid_mask]
    
    return token_level_scores


def split_batch_for_agents(data: DataProto) -> Dict[str, DataProto]:
    agent_roles = data.meta_info['agent_roles']
    new_tensor_batches = {role: {} for role in agent_roles}
    # Keys below contain a role name in the string but are shared labels, not per-role tensors.
    shared_role_named_keys = {"defender_refused"}
    for key in data.batch.keys():
        if key in shared_role_named_keys or key.endswith('_format_reward'):
            for role in agent_roles:
                new_tensor_batches[role][key] = data.batch[key].clone()
            continue
        role_name = ''
        for role in agent_roles:
            if role in key:
                role_name = role
                break
        if role_name in agent_roles:
            v_name = key.replace(f'{role_name}_', '')
            new_tensor_batches[role_name][v_name] = data.batch[key]
        else:
            for role in agent_roles:
                new_tensor_batches[role][key] = data.batch[key].clone()

    for role in agent_roles:
        new_tensor_batches[role]['num_turns'] = torch.tensor(
            data.non_tensor_batch['num_turns'].tolist()
        )
    
    # build non_tensor_batch
    new_non_tensor_batches = {role: {} for role in agent_roles}
    uid_list = data.non_tensor_batch['uid'].tolist()
    for role in agent_roles:
        new_non_tensor_batches[role]['uid'] = np.array(uid_list, dtype=object)
    
    all_agent_batches = {}
    for role in agent_roles:
        all_agent_batches[role] = DataProto.from_dict(new_tensor_batches[role], 
                                                      non_tensors=new_non_tensor_batches[role], 
                                                      meta_info=data.meta_info)

    return all_agent_batches

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayReMASeparatedTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        switch_cfg = config.algorithm.switch_agent
        self.agent_roles = list(switch_cfg.agent_roles) if switch_cfg.agent_roles is not None else []
        if not self.agent_roles:
            raise ValueError("algorithm.switch_agent.agent_roles must not be empty.")
        train_roles_cfg = switch_cfg.get('train_roles', None)
        if train_roles_cfg is None:
            self.trainable_roles = list(self.agent_roles)
        else:
            self.trainable_roles = list(train_roles_cfg)
            invalid_roles = [role for role in self.trainable_roles if role not in self.agent_roles]
            if invalid_roles:
                raise ValueError(f"algorithm.switch_agent.train_roles contains unknown roles: {invalid_roles}")
        if not self.trainable_roles:
            raise ValueError("algorithm.switch_agent.train_roles must contain at least one role.")

        if isinstance(tokenizer, dict):
            self.tokenizers_by_role = tokenizer
        else:
            roles_for_tokenizer = self.agent_roles or ['default']
            self.tokenizers_by_role = {role: tokenizer for role in roles_for_tokenizer}
        missing_roles = [role for role in self.agent_roles if role not in self.tokenizers_by_role]
        if missing_roles:
            raise ValueError(f"Tokenizers missing roles: {missing_roles}")
        default_role = self.agent_roles[-1] if self.agent_roles else next(iter(self.tokenizers_by_role))
        self.tokenizer = self.tokenizers_by_role[default_role]
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self._current_train_agent = None
        self._current_train_agent_idx = None
        self._switch_mode = None
        self._metric_name = None
        self._metric_low = None
        self._metric_high = None
        self._metric_switch_map = None
        self._metric_buffer = None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.Agent0_ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'
            assert Role.Agent1_ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.Agent0_RefPolicy in role_worker_mapping
        self.use_rm = Role.Agent0_RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._init_switch_state()
        if self._switch_mode != 'metric':
            # Build agent switching schedule (defaults to round-robin if update_ratio is not set)
            self._build_switch_schedule()
        
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # Validate switch_agent.update_ratio config if provided
        switch_cfg = config.algorithm.switch_agent
        switch_mode = switch_cfg.get('mode', 'ratio')
        if switch_mode not in ('ratio', 'metric'):
            raise ValueError("algorithm.switch_agent.mode must be 'ratio' or 'metric'.")
        update_ratio_cfg = switch_cfg.get('update_ratio', None)
        if update_ratio_cfg is not None:
            if not hasattr(update_ratio_cfg, 'items'):
                raise ValueError("algorithm.switch_agent.update_ratio must be a mapping from role to positive int.")
            invalid_roles = [role for role in update_ratio_cfg.keys() if role not in self.trainable_roles]
            if invalid_roles:
                raise ValueError(f"algorithm.switch_agent.update_ratio contains unknown roles: {invalid_roles}")
            for role, ratio_val in update_ratio_cfg.items():
                try:
                    ratio_int = int(ratio_val)
                except (TypeError, ValueError):
                    raise ValueError(f"algorithm.switch_agent.update_ratio.{role} must be a positive integer, got {ratio_val!r}")
                if ratio_int <= 0:
                    raise ValueError(f"algorithm.switch_agent.update_ratio.{role} must be positive, got {ratio_int}")
        if switch_mode == 'metric':
            metric_name = switch_cfg.get('metric_name', None)
            if not metric_name:
                raise ValueError("algorithm.switch_agent.metric_name must be set when mode='metric'.")
            metric_low = switch_cfg.get('metric_low', None)
            metric_high = switch_cfg.get('metric_high', None)
            if metric_low is None or metric_high is None:
                raise ValueError("algorithm.switch_agent.metric_low and metric_high must be set when mode='metric'.")
            try:
                metric_low_val = float(metric_low)
                metric_high_val = float(metric_high)
            except (TypeError, ValueError):
                raise ValueError("algorithm.switch_agent.metric_low and metric_high must be numeric.")
            if metric_low_val >= metric_high_val:
                raise ValueError("algorithm.switch_agent.metric_low must be less than metric_high.")
            metric_window = switch_cfg.get('metric_window', 1)
            try:
                metric_window_val = int(metric_window)
            except (TypeError, ValueError):
                raise ValueError("algorithm.switch_agent.metric_window must be a positive integer.")
            if metric_window_val <= 0:
                raise ValueError("algorithm.switch_agent.metric_window must be a positive integer.")
            switch_map = switch_cfg.get('metric_switch_map', None)
            if switch_map is None:
                required_roles = {'attacker', 'defender'}
                if not required_roles.issubset(set(self.trainable_roles)):
                    raise ValueError(
                        "algorithm.switch_agent.metric_switch_map must be set unless train_roles include "
                        "'attacker' and 'defender'."
                    )
            else:
                if not hasattr(switch_map, 'items'):
                    raise ValueError("algorithm.switch_agent.metric_switch_map must be a mapping.")
                invalid_roles = [role for role in switch_map.keys() if role not in self.trainable_roles]
                if invalid_roles:
                    raise ValueError(f"algorithm.switch_agent.metric_switch_map contains unknown roles: {invalid_roles}")
                for role, role_cfg in switch_map.items():
                    if role_cfg is None or not hasattr(role_cfg, 'get'):
                        raise ValueError(
                            f"algorithm.switch_agent.metric_switch_map.{role} must be a mapping with 'above'/'below'."
                        )
                    for bound_key in ('above', 'below'):
                        target_role = role_cfg.get(bound_key, None)
                        if target_role is None:
                            continue
                        if target_role not in self.trainable_roles:
                            raise ValueError(
                                f"algorithm.switch_agent.metric_switch_map.{role}.{bound_key} must be a valid role."
                            )

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        if config.algorithm.filter_groups.enable:
            assert config.actor_rollout_ref.rollout.n > 1

        if config.actor_rollout_ref.actor.clip_mode == 'turn':
            assert config.actor_rollout_ref.actor.agg_mode != 'token'

        if config.reward_model.get('use_format_reward', False) and \
                config.actor_rollout_ref.rollout.max_num_turns > 1:
            print("WARNING: use_format_reward enabled with multi-turn rollout; format rewards apply to the final finished turn.")

        print("[validate_config] All configuration checks passed successfully!")

    def _init_switch_state(self) -> None:
        switch_cfg = self.config.algorithm.switch_agent
        self._switch_mode = switch_cfg.get('mode', 'ratio')
        self._metric_name = switch_cfg.get('metric_name', None)
        self._metric_low = switch_cfg.get('metric_low', None)
        self._metric_high = switch_cfg.get('metric_high', None)
        self._metric_switch_map = switch_cfg.get('metric_switch_map', None)
        self._metric_buffer = None
        if self._switch_mode == 'metric':
            metric_window = int(switch_cfg.get('metric_window', 1))
            self._metric_buffer = deque(maxlen=metric_window)

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = GameDataset(parquet_files=self.config.data.train_files,
                                        #  tokenizer=self.tokenizer,
                                        #  processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                        #  image_key=self.config.data.get('image_key', 'images'),
                                        #  max_prompt_length=self.config.data.max_prompt_length,
                                        #  filter_prompts=True,
                                        #  return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        #  truncation=self.config.data.get('truncation', 'error'),
                                        #  filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                        max_num_turns=self.config.actor_rollout_ref.rollout.max_num_turns
                                        )
        # TODO(ziyu): try to check data in dataset.
        #### UNUSED NOW
        # assert self.train_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = GameDataset(parquet_files=self.config.data.val_files,
                                    #    tokenizer=self.tokenizer,
                                    #    processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       #    image_key=self.config.data.get('image_key', 'images'),
                                       #    max_prompt_length=self.config.data.max_prompt_length,
                                       #    filter_prompts=True,
                                       #    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    #    truncation=self.config.data.get('truncation', 'error'),
                                    #    filter_overlong_prompts=self.config.data.filter_overlong_prompts
                                        max_num_turns=self.config.actor_rollout_ref.rollout.max_num_turns
                                       )
        # TODO(ziyu): try to check data in dataset.     
        ##### UNUSED NOW
        # assert self.val_dataset.truncation == self.config.data.get(
        #     'truncation', 'error'
        # ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        #########################################################
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            # batch_size=len(self.val_dataset),
            batch_size=self.config.data.val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        # assert len(
        #     self.val_dataloader
        # ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores, groundtruths, histories, role_scores=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        if role_scores is None:
            role_scores = [{} for _ in scores]
        samples = list(zip(inputs, outputs, scores, groundtruths, histories, role_scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate_defender_only(self):
        """Validate defender directly on original questions without attacker rewrite."""
        reward_tensor_lst = []
        acc_tensor_lst = []
        data_source_lst = []
        num_turns_lst = []
        completion_tokens_lst = []
        history_lst = []
        sample_groundtruths = []
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        use_prompt = bool(self.config.trainer.get('defender_eval_use_prompt', True))
        prompt_key = self.config.trainer.get('defender_eval_prompt_key', 'prompt')

        def _extract_prompt_text(prompt_messages, fallback_text):
            if isinstance(prompt_messages, np.ndarray):
                prompt_messages = prompt_messages.tolist()
            if isinstance(prompt_messages, dict):
                prompt_messages = [prompt_messages]
            if isinstance(prompt_messages, (list, tuple)):
                for msg in reversed(prompt_messages):
                    if isinstance(msg, dict) and msg.get('role') == 'user' and msg.get('content'):
                        return msg.get('content', '')
                for msg in reversed(prompt_messages):
                    if isinstance(msg, dict) and msg.get('content'):
                        return msg.get('content', '')
            if isinstance(prompt_messages, str):
                return prompt_messages
            return fallback_text or ''

        agent_roles = list(self.config.algorithm.switch_agent.agent_roles)
        if not agent_roles:
            raise ValueError("config.algorithm.switch_agent.agent_roles must not be empty for validation.")
        attacker_role, eval_role = agent_roles[0], agent_roles[-1]
        tokenizer = self.tokenizers_by_role[eval_role]
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for test_data in self.val_dataloader:
            batch_size = len(test_data['question'])
            dummy_tensor = torch.arange(0, batch_size)
            test_data['batch_idx'] = dummy_tensor
            test_batch: DataProto = DataProto.from_single_dict(test_data)

            # Build per-sample dicts for prompt construction
            problem_inputs = []
            for i in range(batch_size):
                problem_inputs.append({k: v[i] for k, v in test_batch.non_tensor_batch.items()})

            # Store ground truths
            ground_truths = [x['ground_truth'] for x in test_batch.non_tensor_batch['reward_model'].tolist()]
            sample_groundtruths.extend(ground_truths)

            chat_lst = None
            attacker_prompt_texts = []
            if use_prompt:
                prompt_messages_batch = []
                has_prompt = True
                for pi in problem_inputs:
                    prompt_messages = pi.get(prompt_key)
                    if isinstance(prompt_messages, np.ndarray):
                        prompt_messages = prompt_messages.tolist()
                    if isinstance(prompt_messages, dict):
                        prompt_messages = [prompt_messages]
                    elif isinstance(prompt_messages, str):
                        prompt_messages = [{'role': 'user', 'content': prompt_messages}]
                    if not prompt_messages:
                        has_prompt = False
                        break
                    prompt_messages_batch.append(prompt_messages)
                    fallback_text = pi.get('seed_prompt') or pi.get('question', '')
                    attacker_prompt_texts.append(_extract_prompt_text(prompt_messages, fallback_text))
                if has_prompt:
                    chat_lst = prompt_messages_batch

            if chat_lst is None:
                # Pretend the attacker just forwards the raw question/seed prompt
                attacker_history = [[{
                    'role': attacker_role,
                    'content': pi.get('seed_prompt') or pi.get('question', ''),
                    'num_gen_tokens': 0,
                    'stop_reason': None,
                }] for pi in problem_inputs]
                attacker_prompt_texts = [hist[0]['content'] for hist in attacker_history]

                # Build defender prompts directly from original questions
                chat_lst = self.multi_agent_rollout._build_chat_list_for_role(
                    role=eval_role,
                    history_list=attacker_history,
                    problem_inputs=problem_inputs,
                    agent_roles=agent_roles,
                )

            sample_inputs.extend(attacker_prompt_texts)
            chat_inputs = self.multi_agent_rollout._apply_chat_template(chat_lst, tokenizer)
            position_ids = compute_position_id_with_mask(chat_inputs["attention_mask"])
            gen_batch = DataProto.from_dict(
                tensors={
                    'input_ids': chat_inputs['input_ids'],
                    'attention_mask': chat_inputs['attention_mask'],
                    'position_ids': position_ids,
                },
                meta_info={
                    'agent_roles': agent_roles,
                    'finish_flag': None,
                    'max_num_turns': 1,
                    'eos_token_id': tokenizer.eos_token_id,
                    'pad_token_id': pad_id,
                    'do_sample': val_kwargs.do_sample,
                    'validate': True,
                },
            )

            # Generate with defender only
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg[eval_role].world_size)
            gen_output_padded = self.actor_rollout_wg[eval_role].generate_sequences(gen_batch_padded)
            gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)

            output_ids = gen_output.batch['responses']
            # Count non-pad tokens as completion length
            completion_tokens = (output_ids != pad_id).sum(dim=-1).cpu()
            completion_tokens_lst.append(completion_tokens)
            output_texts = [
                tokenizer.decode(ids[:int(l.item())], skip_special_tokens=True)
                for ids, l in zip(output_ids, completion_tokens)
            ]
            sample_outputs.extend(output_texts)

            # Prepare history with attacker raw question and defender response
            histories = []
            for i in range(batch_size):
                histories.append([
                    {
                        'role': attacker_role,
                        'content': attacker_prompt_texts[i],
                        'num_gen_tokens': 0,
                        'stop_reason': None,
                    },
                    {
                        'role': eval_role,
                        'content': output_texts[i],
                        'num_gen_tokens': int(completion_tokens[i].item()),
                        'stop_reason': 'stop',
                    },
                ])
            history_arr = np.array(histories, dtype=object)
            history_lst.append(history_arr)

            # Attach auxiliary tensors expected by reward manager
            for role in agent_roles:
                if role == eval_role:
                    gen_output.batch[f'{role}_num_gen_tokens'] = completion_tokens.clone()
                else:
                    gen_output.batch[f'{role}_num_gen_tokens'] = torch.zeros_like(completion_tokens)
                gen_output.batch[f'{role}_turn_finished'] = torch.zeros_like(completion_tokens)

            gen_output.non_tensor_batch.update({
                'history': history_arr,
                'num_turns': np.ones(batch_size, dtype=int),
                'response': np.array(output_texts, dtype=object),
                'data_source': test_batch.non_tensor_batch.get('data_source', np.array(['unknown'] * batch_size, dtype=object)),
            })
            # Preserve original non-tensor fields
            gen_output.non_tensor_batch.update(test_batch.non_tensor_batch)
            format_reward_roles = self.config.reward_model.get('format_reward_roles', None)
            if format_reward_roles is not None:
                format_reward_roles = list(format_reward_roles)
            gen_output.meta_info.update({
                'mask_unfinished_reward': self.config.reward_model.mask_unfinished_reward,
                'use_format_reward': self.config.reward_model.get('use_format_reward', False),
                'format_reward_roles': format_reward_roles,
            })

            test_batch = test_batch.union(gen_output)

            reward_tensor = self.val_reward_fn(test_batch)
            reward_tensor_lst.append(reward_tensor[f'{eval_role}_turn_level_reward'])
            acc_tensor_lst.append(reward_tensor['acc'])

            scores = reward_tensor[f'{eval_role}_turn_level_reward'].sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            num_turns_lst.append(torch.ones(batch_size, dtype=torch.float32))
            data_sources = test_batch.non_tensor_batch.get('subset', test_batch.non_tensor_batch.get(
                'data_source', ['unknown'] * reward_tensor[f'{eval_role}_turn_level_reward'].shape[0]))
            data_source_lst.append(data_sources)

        flat_histories = []
        for history_arr in history_lst:
            flat_histories.extend(history_arr.tolist())

        self._maybe_log_val_generations(inputs=sample_inputs,
                                        outputs=sample_outputs,
                                        scores=sample_scores,
                                        role_scores=None,
                                        groundtruths=sample_groundtruths,
                                        histories=flat_histories)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        metric_dict = {}
        for data_source, rewards in self._group_by_data_source(data_sources, reward_tensor):
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in self._group_by_data_source(data_sources, acc_tensor):
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)

        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['val/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['val/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['val/num_turns/min'] = num_turns_tensor.min().item()

        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['val/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['val/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['val/completion_tokens/min'] = completion_tokens_tensor.min().item()

        return metric_dict

    @staticmethod
    def _group_by_data_source(data_sources, values_tensor):
        data_source_map = {}
        for i in range(len(values_tensor)):
            ds = data_sources[i]
            if ds not in data_source_map:
                data_source_map[ds] = []
            data_source_map[ds].append(values_tensor[i].item())
        return data_source_map.items()

    def _validate(self):
        reward_tensor_lst = []
        acc_tensor_lst = []
        data_source_lst = []
        num_turns_lst = []
        history_lst = []
        sample_groundtruths = []
        completion_tokens_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_role_scores = []

        if self.config.trainer.get('defender_only_eval', True):
            return self._validate_defender_only()

        agent_roles = list(self.config.algorithm.switch_agent.agent_roles)
        if not agent_roles:
            raise ValueError("config.algorithm.switch_agent.agent_roles must not be empty for validation.")
        eval_role = agent_roles[-1]
        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        rollout_meta_info = {
            'agent_roles': agent_roles,
            'finish_flag': None,
            'max_num_turns': max_num_turns,
            'train_roles': self.trainable_roles,
        }

        for test_data in self.val_dataloader:
            # test_batch = DataProto.from_single_dict(test_data)
            dummy_tensor = torch.arange(0, len(test_data['question']))
            test_data['batch_idx'] = dummy_tensor
            test_batch: DataProto = DataProto.from_single_dict(test_data, meta_info=rollout_meta_info)          

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            reward_style = test_batch[0].non_tensor_batch['reward_model']['style']
            if self.config.reward_model.enable and reward_style == 'model':
                return {}

            # Store original inputs
            # input_ids = test_batch.batch['input_ids']
            # input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            input_texts = test_batch.non_tensor_batch['question']
            sample_inputs.extend(input_texts)

            # Store original ground truth if available
            ground_truths = [x['ground_truth'] for x in test_data['reward_model'].tolist()]

            sample_groundtruths.extend(ground_truths)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                raise NotImplementedError('validation is not implemented yet')
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.select(
                        batch_keys=['batch_idx'], 
                        non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch.keys() 
                        if k not in ['data_source', 'ability', 'reward_model', 'extra_info', 'uid']], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts'], 
                        deepcopy=True
                    )
            
            test_gen_batch.meta_info.update({
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            })
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size

            first_role = agent_roles[0]
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg[first_role].world_size)
            test_output_gen_batch_padded = self.multi_turn_generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            # output_ids = test_output_gen_batch.batch['responses']
            # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            output_texts = test_output_gen_batch.non_tensor_batch['response']
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
            test_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
            format_reward_roles = self.config.reward_model.get('format_reward_roles', None)
            if format_reward_roles is not None:
                format_reward_roles = list(format_reward_roles)
            test_batch.meta_info['format_reward_roles'] = format_reward_roles
            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)
            turn_reward_key = f'{eval_role}_turn_level_reward'
            if turn_reward_key not in reward_tensor:
                fallback_keys = [k for k in reward_tensor.keys() if k.endswith('_turn_level_reward')]
                if not fallback_keys:
                    raise KeyError(f'Validation reward tensor does not contain turn-level rewards. Keys: {list(reward_tensor.keys())}')
                turn_reward_key = fallback_keys[-1]
            reward_tensor_lst.append(reward_tensor[turn_reward_key])
            acc_tensor_lst.append(reward_tensor['acc'])

            # Store scores
            scores = reward_tensor[turn_reward_key].sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            # per-role summed turn rewards
            per_role_scores = []
            for i in range(reward_tensor[turn_reward_key].shape[0]):
                role_score_dict = {}
                for role in agent_roles:
                    key = f'{role}_turn_level_reward'
                    if key in reward_tensor:
                        role_score_dict[role] = reward_tensor[key][i].sum().item()
                per_role_scores.append(role_score_dict)
            sample_role_scores.extend(per_role_scores)
            num_turns = torch.tensor(test_output_gen_batch.non_tensor_batch['num_turns'].tolist(), dtype=torch.float32, device="cpu")
            num_turns_lst.append(num_turns)
            role_token_tensors = []
            for role in agent_roles:
                role_key = f'{role}_num_gen_tokens'
                if role_key in test_output_gen_batch.batch.keys():
                    role_token_tensors.append(test_output_gen_batch.batch[role_key].cpu())
            if role_token_tensors:
                turn_level_completion_tokens = torch.stack(role_token_tensors, dim=0).sum(dim=0)
                completion_tokens = turn_level_completion_tokens.sum(dim=-1)
                completion_tokens_lst.append(completion_tokens)

            # not use `data_source`, use `subset` instead
            data_source_lst.append(test_batch.non_tensor_batch.get(
                'subset',
                test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor[turn_reward_key].shape[0]),
            ))
            
            history_lst.append(test_output_gen_batch.non_tensor_batch['history'].tolist())

        flat_histories = []
        for history_batch in history_lst:
            flat_histories.extend(history_batch)

        self._maybe_log_val_generations(inputs=sample_inputs,
                                        outputs=sample_outputs,
                                        scores=sample_scores,
                                        role_scores=sample_role_scores,
                                        groundtruths=sample_groundtruths,
                                        histories=flat_histories)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu() #(batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        metric_dict = {}
        for data_source, rewards in self._group_by_data_source(data_sources, reward_tensor):
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in self._group_by_data_source(data_sources, acc_tensor):
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)
        
        # Add num_turns and completion_tokens metrics
        if num_turns_lst:
            num_turns_tensor = torch.cat(num_turns_lst, dim=0)
            metric_dict['val/num_turns/mean'] = num_turns_tensor.float().mean().item()
            metric_dict['val/num_turns/max'] = num_turns_tensor.max().item()
            metric_dict['val/num_turns/min'] = num_turns_tensor.min().item()
        
        if completion_tokens_lst:
            completion_tokens_tensor = torch.cat(completion_tokens_lst, dim=0)
            metric_dict['val/completion_tokens/mean'] = completion_tokens_tensor.float().mean().item()
            metric_dict['val/completion_tokens/max'] = completion_tokens_tensor.max().item()
            metric_dict['val/completion_tokens/min'] = completion_tokens_tensor.min().item()

        # Save generation results to a JSON file
        if self.config.trainer.get('save_val_generations', False):
            output_dir = Path(self.config.trainer.default_local_dir) / 'eval_records'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'val_step_{self.global_steps}.jsonl'
            
            # Concatenate history lists from different batches
            all_histories = []
            for history_batch in history_lst:
                all_histories.extend(history_batch)
            
            results_to_save = []
            for inp, outp, gt, hist, score, role_score in zip(sample_inputs, sample_outputs, sample_groundtruths, all_histories, sample_scores, sample_role_scores):
                unpad_history = [x for x in hist if x['role'] != 'padding']
                # XXX(ziyu): Maybe should develop a better way to store any tasks input,
                #  currently not saving reponse_a/b for ReMA-laaj
                results_to_save.append({
                    'question': inp,
                    'answer': outp, 
                    'groundtruth': gt,
                    'history': unpad_history,
                    'score': score,
                    'role_scores': role_score,
                })
            
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(results_to_save)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent0_ActorRollout)
            agent0_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent0_config.model.path = self.config.algorithm.switch_agent.model_paths[0]
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Agent0_ActorRollout],
                                                     config=agent0_config,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['agent0_actor_rollout'] = actor_rollout_cls

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Agent1_ActorRollout)
            agent1_config = copy.deepcopy(self.config.actor_rollout_ref)
            agent1_config.model.path = self.config.algorithm.switch_agent.model_paths[1]
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Agent1_ActorRollout],
                                                     config=agent1_config,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['agent1_actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            raise NotImplementedError
            # resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            raise NotImplementedError
            # resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
            #                                       config=self.config.actor_rollout_ref,
            #                                       role='ref')
            # self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg0 = all_wg['agent0_actor_rollout']
        self.actor_rollout_wg0.init_model()
        
        self.actor_rollout_wg1 = all_wg['agent1_actor_rollout']
        self.actor_rollout_wg1.init_model()
        
        if len(self.agent_roles) != 2:
            raise ValueError("Current trainer expects exactly two agent roles for actor rollout workers.")
        self.actor_rollout_wg = {
            self.agent_roles[0]: self.actor_rollout_wg0,
            self.agent_roles[1]: self.actor_rollout_wg1,
        }

        self.multi_agent_rollout = MultiAgentRollout(
            self.config.actor_rollout_ref.rollout,
            self.tokenizers_by_role,
            self.actor_rollout_wg,
        )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        import gc; gc.collect()
        for role, wg in self.actor_rollout_wg.items():
            actor_local_path = os.path.join(local_global_step_folder, f'{role}/actor')

            actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', f'{role}/actor')
            wg.save_checkpoint(actor_local_path,
                               actor_remote_path,
                               self.global_steps,
                               remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            for role, wg in self.critic_wg:
                critic_local_path = os.path.join(local_global_step_folder, f'{role}/critic')
                critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                    self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', f'{role}/critic')
                wg.save_checkpoint(critic_local_path,
                                            critic_remote_path,
                                            self.global_steps,
                                            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        # load actor
        for role, wg in self.actor_rollout_wg.items():
            actor_path = os.path.join(global_step_folder, f'{role}/actor')
            wg.load_checkpoint(actor_path,
                                del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            for role, wg in self.critic_wg.items():
                critic_path = os.path.join(global_step_folder, f'{role}/critic')
                wg.load_checkpoint(critic_path,
                                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        # meta_thinking_attention_mask = batch.batch['meta_thinking_attention_mask']
        # reasoning_attention_mask = batch.batch['reasoning_attention_mask']
        batch_size = attention_mask.shape[0]
        # global_seqlen_lst = (meta_thinking_attention_mask.view(batch_size, -1).sum(-1) + reasoning_attention_mask.view(batch_size, -1).sum(-1)).tolist()  # (train_batch_size,)
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg[self._current_train_agent].world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
        
    
    def multi_turn_generate_sequences(self, gen_batch: DataProto):
        agent_roles = gen_batch.meta_info['agent_roles']
        dummy_batch = DataProto.from_dict(
            {'dummy_tensor': torch.arange(0, len(gen_batch.batch))}
        )
        for agent_role, wg in self.actor_rollout_wg.items():
            assert agent_role in agent_roles, f'{agent_roles=}, {agent_role=}'
            wg.enter_generate_context(dummy_batch)            
        
        
        try: 
            output = self.multi_agent_rollout.generate(gen_batch)
        except Exception as e:
            raise e
        finally:
            for agent_role, wg in self.actor_rollout_wg.items():
                wg.exit_generate_context(dummy_batch)      
    
        return output

    def _build_switch_schedule(self) -> None:
        """Build a cyclic schedule for training agents based on update_ratio."""
        switch_config = self.config.algorithm.switch_agent
        update_ratio_cfg = switch_config.get('update_ratio', None)

        ratios = {role: 1 for role in self.trainable_roles}
        if update_ratio_cfg is not None:
            for role, ratio_val in update_ratio_cfg.items():
                if role not in ratios:
                    # This should have been caught during validation; keep going defensively.
                    continue
                ratio_int = int(ratio_val)
                ratios[role] = ratio_int

        schedule: List[str] = []
        for role in self.trainable_roles:
            schedule.extend([role] * ratios[role])

        if not schedule:
            raise ValueError("Switch schedule is empty. Check train_roles or update_ratio.")

        start_agent = switch_config.start_agent
        if start_agent not in self.trainable_roles:
            start_agent = self.trainable_roles[0]
        start_pos = schedule.index(start_agent)
        if start_pos:
            schedule = schedule[start_pos:] + schedule[:start_pos]

        self._switch_schedule = schedule

    def _update_current_train_agent(self, epoch: int = None, metrics: Optional[Dict[str, float]] = None) -> None:
        """Update the current training agent based on switch config.
        
        Args:
            epoch (int, optional): Current epoch number. Only needed for epoch-level switching.
        """
        switch_config = self.config.algorithm.switch_agent
        if self._switch_mode == 'metric':
            if self._current_train_agent is None:
                start_agent = switch_config.start_agent
                if start_agent not in self.trainable_roles:
                    start_agent = self.trainable_roles[0]
                self._current_train_agent = start_agent
                self._current_train_agent_idx = self.trainable_roles.index(start_agent)
            if metrics is not None:
                self._maybe_switch_by_metric(metrics)
            return

        schedule = getattr(self, '_switch_schedule', [])
        if not schedule:
            self._build_switch_schedule()
            schedule = self._switch_schedule
        if not schedule:
            raise ValueError("No trainable roles configured for switch_agent.")

        freq = max(1, switch_config.freq)

        if switch_config.level == 'step':
            progress_units = self.global_steps // freq
        elif switch_config.level == 'epoch':
            if epoch is None:
                epoch_idx = self.global_steps // max(1, len(self.train_dataloader))
            else:
                epoch_idx = epoch
            progress_units = epoch_idx // freq
        else:
            raise ValueError(f"Unknown switch_level: {switch_config.level}")

        self._current_train_agent_idx = progress_units % len(schedule)
        new_agent = schedule[self._current_train_agent_idx]
        if self._current_train_agent != new_agent:
            print(f'Training switching to {new_agent}')
            self._current_train_agent = new_agent

    def _maybe_switch_by_metric(self, metrics: Dict[str, float]) -> None:
        if self._metric_buffer is None or not self._metric_name:
            return
        if not metrics or self._metric_name not in metrics:
            return
        try:
            metric_value = float(metrics[self._metric_name])
        except (TypeError, ValueError):
            return
        if not math.isfinite(metric_value):
            return
        self._metric_buffer.append(metric_value)
        metric_avg = sum(self._metric_buffer) / len(self._metric_buffer)
        target = self._get_metric_switch_target(self._current_train_agent, metric_avg)
        if target and target != self._current_train_agent:
            print(f"Training switching to {target} (metric {self._metric_name}={metric_avg:.4f})")
            self._current_train_agent = target
            self._current_train_agent_idx = self.trainable_roles.index(target)

    def _get_metric_switch_target(self, current_role: Optional[str], metric_value: float) -> Optional[str]:
        if current_role is None:
            return None
        low = float(self._metric_low)
        high = float(self._metric_high)
        if self._metric_switch_map:
            role_cfg = self._metric_switch_map.get(current_role, {})
            if metric_value >= high:
                return role_cfg.get('above', None)
            if metric_value <= low:
                return role_cfg.get('below', None)
            return None
        if current_role == 'defender' and metric_value >= high:
            return 'attacker'
        if current_role == 'attacker' and metric_value <= low:
            return 'defender'
        return None

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        if self.config.trainer.get('fork_wandb_id', None) is not None:
            fork_wandb_id = self.config.trainer.fork_wandb_id
            # wandb_kwargs = {'resume': 'must', 'id': fork_wandb_id}
            print(f'**[WANDB]: will fork run from wandb id: `{fork_wandb_id}` at step {self.global_steps} **')
            
            # e.g. fork_from="6yaq69uw?_step=200"
            wandb_kwargs = {'fork_from': f"{fork_wandb_id}?_step={self.global_steps}"}
        else:
            wandb_kwargs = {}
        
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_kwargs=wandb_kwargs
                          )

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        self._update_current_train_agent()
        print(f'Starting training with {self._current_train_agent}')

        max_num_turns = self.config.actor_rollout_ref.rollout.max_num_turns
        base_rollout_meta_info = {
            'agent_roles': self.agent_roles,
            'finish_flag': None,
            'max_num_turns': max_num_turns,
            'train_roles': self.trainable_roles,
        }
        
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        total_prompt_cnt = 0 
        all_negative_cnt = 0
        all_positive_cnt = 0

        for epoch in range(self.config.trainer.total_epochs):
            self._update_current_train_agent(epoch=epoch)
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # create a dummy tensor for the construction function
                dummy_tensor = torch.arange(0, len(batch_dict['question']))
                batch_dict['batch_idx'] = dummy_tensor
                current_meta_info = dict(base_rollout_meta_info)
                current_meta_info['train_role'] = self._current_train_agent
                new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=current_meta_info)
                new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                num_gen_batches += 1

                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    raise NotImplementedError('multi_modal_inputs is not implemented yet')
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    # because verl originally calls this 'chat'
                    gen_batch = new_batch.select(
                        batch_keys=['batch_idx'], 
                        non_tensor_batch_keys=[k for k in new_batch.non_tensor_batch.keys() 
                        if k not in ['data_source', 'ability', 'reward_model', 'extra_info', 'uid']], 
                        meta_info_keys=['agent_roles', 'finish_flag', 'system_prompts'], 
                        deepcopy=True
                    )
                    gen_batch.meta_info['train_role'] = self._current_train_agent

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.multi_turn_generate_sequences(gen_batch)
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError('REMAX is not implemented yet')
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    print(f"[DEBUG] global_step={self.global_steps} merged attacker/defender outputs")

                    
                    # compute global_valid tokens
                    global_attention = None
                    for role in new_batch.meta_info['agent_roles']:
                        attn_key = f'{role}_attention_mask'
                        if attn_key in new_batch.batch:
                            role_mask = new_batch.batch[attn_key]
                            global_attention = role_mask if global_attention is None else global_attention + role_mask
                    if global_attention is None:
                        raise KeyError('No role attention masks found when computing global_token_num')
                    new_batch.meta_info['global_token_num'] = torch.sum(
                        global_attention,
                        dim=-1
                    ).tolist()

                    # # recompute old_log_probs
                    # with _timer('old_log_prob', timing_raw):
                    #     old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    #     batch = batch.union(old_log_prob)


                    # compute values
                    if self.use_critic:
                        raise NotImplementedError('critic is not implemented yet')
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('reward', timing_raw):
                        # print(f"[DEBUG] global_step={self.global_steps} start reward_fn")
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            raise NotImplementedError('RM is not implemented yet')
                            # # we first compute reward model score
                            # reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # batch = batch.union(reward_tensor)

                        # add mask_unfinished_reward to meta_info
                        new_batch.meta_info['mask_unfinished_reward'] = self.config.reward_model.mask_unfinished_reward
                        new_batch.meta_info['use_format_reward'] = self.config.reward_model.get('use_format_reward', False)
                        format_reward_roles = self.config.reward_model.get('format_reward_roles', None)
                        if format_reward_roles is not None:
                            format_reward_roles = list(format_reward_roles)
                        new_batch.meta_info['format_reward_roles'] = format_reward_roles
                        # rule-based rm build token-level reward_tensor_map for each agent
                        # {
                        #     "attacker_turn_level_reward": tensor([...], device='cuda:0'),
                        #     "defender_turn_level_reward": tensor([...], device='cuda:0'),
                        # }
                        reward_tensor_map = self.reward_fn(new_batch)
                        # print(f"[DEBUG] global_step={self.global_steps} finish reward_fn")
                        # batch.batch['token_level_scores'] = reward_tensor
                        new_batch.batch['acc'] = reward_tensor_map.pop('acc')
                        for key_reward, reward_tensor in reward_tensor_map.items():
                            new_batch.batch[key_reward] = reward_tensor
                            # Only compute returns for turn-level rewards; scalar scores have dim < 2.
                            if reward_tensor.dim() < 2 or not key_reward.endswith('_turn_level_reward'):
                                continue
                            # get_turn_mask, shape (bsz, max_num_turns), 1 for valid turn, 0 for invalid turn
                            turn_mask = verl_F.get_turn_mask(reward_tensor, new_batch.non_tensor_batch['num_turns'])
                            key_return = key_reward.replace('reward', 'return')
                            # compute turn_level return with turn_level_gamma
                            new_batch.batch[key_return] = core_algos.compute_turn_level_return(
                                reward_tensor, turn_mask, self.config.algorithm.gamma_turn_level)
                    
                    # statistics for group filter
                    if self.config.actor_rollout_ref.rollout.n > 1:
                        # key_reward = list(reward_tensor_map.keys())[0]
                        # one_agent_reward_tensor = reward_tensor_map[key_reward]
                        acc_tensor = new_batch.batch['acc']
                        id2acc = defaultdict(list)
                        for i_bsz, uid in enumerate(new_batch.non_tensor_batch['uid']):
                            id2acc[uid].append(acc_tensor[i_bsz])

                        kept_prompt_uids = []
                        for key_uid, acc_this_uid in id2acc.items():
                            acc_this_uid = torch.tensor(acc_this_uid)
                            if (acc_this_uid == 0).all():
                                all_negative_cnt += 1
                            elif (acc_this_uid == 1).all():
                                all_positive_cnt += 1
                            else:
                                # keep prompt with none-zero advantages
                                kept_prompt_uids.append(key_uid)
                            total_prompt_cnt += 1
                    
                    if not self.config.algorithm.filter_groups.enable:
                        # if not enable group filter, keep all data
                        batch = new_batch
                    else:
                        # filter data based on group filter statistics
                        num_prompt_in_batch += len(kept_prompt_uids)
                        # get kept data batch
                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])
                        
                        # check if we have enough data
                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            # keep generating
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    if self.config.actor_rollout_ref.rollout.n > 1:
                        metrics.update({
                            'rollout/all_negative_cnt': all_negative_cnt,
                            'rollout/all_positive_cnt': all_positive_cnt,
                            'rollout/total_prompt_cnt': total_prompt_cnt,
                            'rollout/num_gen_batches': num_gen_batches,
                        })
                    
                    with _timer('save_train_generation', timing_raw):
                        # save train generation
                        if self.config.trainer.get('save_train_generations', False):
                            self._save_train_generations(batch)

                    

                    with _timer('adv', timing_raw):
                        print(f"[DEBUG] global_step={self.global_steps} start advantage, agent={self._current_train_agent}")
                        # Merge different role data into a single DataProto
                        agents_batches: Dict[str, DataProto] = split_batch_for_agents(batch)
                        agent_batch = agents_batches[self._current_train_agent]
                        
                        # assign turn_level scores to the last token of each turn, w/ step_ids
                        #  and then i'll call compute_advantage to distribute the score to all
                        #  tokens of each step.
                        token_level_scores = compute_token_level_scores(agent_batch)
                        agent_batch.batch['token_level_scores'] = token_level_scores
                        batch = agent_batch
                        
                        # # compute rewards. apply_kl_penalty if available
                        # if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                          kl_ctrl=self.kl_ctrl,
                        #                                          kl_penalty=self.config.algorithm.kl_penalty)
                        #     metrics.update(kl_metrics)
                        # else:
                        #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        
                        # XXX(ziyu): debug
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # in this case, its usage is changed.
                        # for REINFORCE++, it's used to distribute the score from last token of each turn
                        # to all tokens of each step.
                        # for GRPO, we use turn_level_reward.sum(-1) as the outcome reward and then
                        # assign each label token the normalized advantage.
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma_token_level,
                                                  lam=self.config.algorithm.lam_token_level,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        print(f"[DEBUG] global_step={self.global_steps} finish advantage")

                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        print(f"[DEBUG] global_step={self.global_steps} start compute_log_prob agent={self._current_train_agent}")
                        old_log_prob = self.actor_rollout_wg[self._current_train_agent].compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                        print(f"[DEBUG] global_step={self.global_steps} finish compute_log_prob")

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg[self._current_train_agent].compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)


                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg[self._current_train_agent].update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            print(f"[DEBUG] global_step={self.global_steps} start update_actor agent={self._current_train_agent}")
                            actor_output = self.actor_rollout_wg[self._current_train_agent].update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        print(f"[DEBUG] global_step={self.global_steps} finish update_actor agent={self._current_train_agent}")

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                metrics.update({
                    'train/current_agent_idx': self._current_train_agent_idx,
                    'train/current_agent': self._current_train_agent,
                })
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                all_negative_cnt = 0
                all_positive_cnt = 0
                total_prompt_cnt = 0

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1
                self._update_current_train_agent(metrics=metrics)

    def _save_train_generations(self, batch: DataProto):
        # save train generations
        output_dir = Path(self.config.trainer.default_local_dir) / 'replay_buffer'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'train_step_{self.global_steps}.jsonl'

        agent_roles = batch.meta_info.get('agent_roles', []) if batch.meta_info else []
        results_dict = {}
        for i, data_item in enumerate(batch):
            uid = data_item.non_tensor_batch['uid']
            if uid not in results_dict:
                results_dict[uid] = {
                    "question": data_item.non_tensor_batch['question'],
                    "groundtruth": data_item.non_tensor_batch['reward_model']['ground_truth'],
                    "response": [],
                    "history": [],
                    "score": [],
                    "role_scores": [],
                    "latest_role_outputs": [],
                    "finish_reason": [],
                }
                if data_item.non_tensor_batch['data_source'] == "ReMA-laaj":
                    results_dict[uid]['response_a'] = data_item.non_tensor_batch['response_a']
                    results_dict[uid]['response_b'] = data_item.non_tensor_batch['response_b']
                results_dict[uid]['reward_components'] = []

            padded_history = data_item.non_tensor_batch['history']
            valid_history = [x for x in padded_history if x['role'] != 'padding']
            results_dict[uid]['history'].append(valid_history)
            results_dict[uid]['response'].append(data_item.non_tensor_batch['response'])

            role_scores = {}
            for role in agent_roles:
                key = f'{role}_turn_level_reward'
                if key in data_item.batch:
                    role_scores[role] = data_item.batch[key].sum().item()
            results_dict[uid]['role_scores'].append(role_scores)
            total_score = sum(role_scores.values()) if role_scores else 0.0
            results_dict[uid]['score'].append(total_score)
            safety_reward = float(data_item.batch['acc']) if 'acc' in data_item.batch else None
            revision_reward = float(data_item.batch['revision_scores']) if 'revision_scores' in data_item.batch else None
            defender_quality_reward = float(data_item.batch['defender_quality_scores']) if 'defender_quality_scores' in data_item.batch else None

            def _last_role_msg(role_name):
                for message in reversed(valid_history):
                    if message.get('role') == role_name:
                        return message
                return None

            latest_outputs = {}
            for role in agent_roles:
                last_msg = _last_role_msg(role)
                if last_msg:
                    latest_outputs[role] = {
                        'content': last_msg.get('content'),
                        'parsed_answer': last_msg.get('parsed_answer'),
                        'parsed_think': last_msg.get('parsed_think'),
                    }
            if latest_outputs:
                results_dict[uid]['latest_role_outputs'].append(latest_outputs)

            results_dict[uid]['finish_reason'].append(
                data_item.non_tensor_batch['finish_reason']
            )

            # reward breakdown for logging
            attacker_msg = _last_role_msg('attacker')
            defender_msg = _last_role_msg('defender')
            if attacker_msg:
                attacker_prompt = attacker_msg.get('content') or ""
                if USE_ANSWER_TAGS_FOR_REWARD_MODEL:
                    attacker_prompt = attacker_msg.get('parsed_answer') or attacker_prompt
            else:
                attacker_prompt = data_item.non_tensor_batch.get('seed_prompt', '')
            if defender_msg:
                defender_response = defender_msg.get('content') or ""
                if USE_ANSWER_TAGS_FOR_REWARD_MODEL:
                    defender_response = defender_msg.get('parsed_answer') or defender_response
            else:
                defender_response = data_item.non_tensor_batch.get('response', '')
            harm_reward = None
            refusal_reward = None
            labels = None
            try:
                harm_reward, refusal_reward, labels = compute_score_components(
                    data_source=data_item.non_tensor_batch.get('data_source', 'game'),
                    attacker_prompt=attacker_prompt or "",
                    defender_response=defender_response or "",
                    extra_info=data_item.non_tensor_batch.get('extra_info', {}),
                )
            except Exception as e:
                # best-effort logging; continue on error
                labels = {"error": str(e)}

            format_reward = None
            if defender_msg:
                try:
                    format_reward = format_reward_func(defender_msg.get('content', ''))
                except Exception:
                    pass

            results_dict[uid]['reward_components'].append({
                "safety": safety_reward,
                "revision": revision_reward,
                "defender_quality": defender_quality_reward,
                "format": format_reward,
                "harm": harm_reward,
                "refusal": refusal_reward,
                "labels": labels,
            })

        results_to_save = []
        for uid, result in results_dict.items():
            result['avg_score'] = sum(result['score']) / len(result['score'])
            results_to_save.append(result)
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(results_to_save)
