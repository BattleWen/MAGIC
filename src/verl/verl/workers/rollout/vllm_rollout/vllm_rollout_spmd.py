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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.protocol import collate_fn as data_proto_collate_fn
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.utils.model import compute_position_id_with_mask
from transformers import PreTrainedTokenizer
from string import Formatter

from verl.utils.reward_score.game import extract_answer, extract_think

from verl.utils.reward_score.game import extract_answer, extract_think

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics
def _get_format_str_keys(format_str: str) -> List[str]:
    formatter = Formatter()
    keys = [fname for _, fname, _, _ in formatter.parse(format_str) if fname]
    return keys

def _fill_format_str(format_str: str, input_data: Dict[str, str]):
    target_keys = _get_format_str_keys(format_str)
    # required_data = {k: v for k, v in input.items() if k in target_keys}
    required_data = dict()
    for k in target_keys:
        assert k in input_data.keys(), (k, input_data.keys())
        required_data[k] = input_data[k]
    return format_str.format(**required_data)

def _reshape_dict(data_dict: Dict[str, List[Any]], batch_size: int)->List[Dict[str, Any]]:
    ret = [dict() for _ in range(batch_size)]
    for k, v_list in data_dict.items():
        for i_bsz in range(batch_size):
            ret[i_bsz][k] = v_list[i_bsz]
    return ret

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id,
                        prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id,
                                  as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray],
                       repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer,
                 model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (
            not config.enforce_eager and config.free_cache_engine
        ), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert (
            tensor_parallel_size <= torch.distributed.get_world_size()
        ), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens",
                                                 8192)

        if kwargs.get("train_tp", None) is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(
                tensor_model_parallel_size=tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        assert (
            model_hf_config.max_position_embeddings
            >= config.prompt_length + config.response_length
        ), "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.get("enable_sleep_mode", True),
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            seed=config.get('seed', 0)
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
                and self.config.free_cache_engine):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ],
                dtype=object,
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"),
                    non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append({
                    "prompt_token_ids": raw_prompt_ids,
                    "multi_modal_data": multi_modal_data,
                })
        else:
            vllm_inputs = [{
                "prompt_token_ids": raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

            for i in range(len(vllm_inputs)):
                if isinstance(vllm_inputs[i]["prompt_token_ids"], np.ndarray):
                    vllm_inputs[i]["prompt_token_ids"] = vllm_inputs[i][
                        "prompt_token_ids"].tolist()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        is_multi_turn = prompts.meta_info.get("is_multi_turn", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        if is_multi_turn:
            kwargs.update({
                "n": 1, # if is_multi_turn, already repeat in ray_trainer
                "include_stop_str_in_output": True,
                "detokenize": True,
            })
            if prompts.meta_info.get('finish_flag') is not None:
                kwargs['stop'] = [prompts.meta_info['finish_flag']]


        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=
                vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            stop_reasons = []
            gen_response_lengths = []
            text = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    stop_reasons.append(output.outputs[sample_id].finish_reason)
                    gen_response_lengths.append(len(output.outputs[sample_id].token_ids))
                    text.append(output.outputs[sample_id].text)
            
            non_tensor_batch["gen_response_lengths"] = np.array(gen_response_lengths, dtype=object)
            non_tensor_batch["stop_reasons"] = np.array(stop_reasons, dtype=object)
            non_tensor_batch["text"] = np.array(text, dtype=object)
            response = pad_2d_list_to_length(
                response,
                self.pad_token_id,
                max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask,
                                                    self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids,
                                                  self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    raise NotImplementedError("multi_modal_inputs is not supported for multi-turn generation")
                    non_tensor_batch[
                        "multi_modal_inputs"] = _repeat_interleave(
                            non_tensor_batch["multi_modal_inputs"],
                            self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1,
                                         response_length + 1,
                                         device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(
            batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1,
                                                       -1).expand(
                                                           batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response,
                                               eos_token=eos_token_id,
                                               dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask),
                                   dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (vllm_version in ("0.3.1", "0.4.2", "0.5.4", "0.6.3")
                and self.config.free_cache_engine):
            self.inference_engine.free_cache_engine()

        # Preserve prompt meta_info so downstream consumers (e.g., reward functions)
        # continue to see fields like max_num_turns.
        return DataProto(batch=batch,
                         non_tensor_batch=non_tensor_batch,
                         meta_info=prompts.meta_info)

    @torch.no_grad()
    def multi_turn_generate_sequences(
        self,
        prompts: DataProto,
        tokenizer: PreTrainedTokenizer,
        max_num_turns: int,
        agent_roles: List[str],
        finish_flag: Optional[str],
        **kwargs,
    ) -> DataProto:
        """Main function responsible for coordinating multi-turn dialogue generation"""
        # Use the parameters directly, assuming they are correctly defined
        # add extra code for multi-turn generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        prompts.meta_info['is_multi_turn'] = True
        prompts.meta_info['finish_flag'] = finish_flag

        batch_size = len(prompts.non_tensor_batch['index'])
        # Initialize state variables
        history, finish_flags, finish_reason = self._initialize_conversation_state(
            batch_size
        )

        # Multi-turn dialogue generation
        # this will change the history, finish_flags, finish_reason
        latest_outputs, conversation_history = self._run_multi_turn_conversation(
            prompts,
            tokenizer=tokenizer,
            max_num_turns=max_num_turns,
            agent_roles=agent_roles,
            finish_flag=finish_flag,
            history=history,
            finish_flags=finish_flags,
            finish_reason=finish_reason,
            response_length=self.config.response_length,
            **kwargs,
        )

        # Mark completion reasons
        # this will change the finish_reason
        if max_num_turns > 1:
            self._mark_unfinished_as_max_turns(finish_flags, finish_reason)

        last_round_responses = [{
            m['role']: m['content']
            for m in h[-2:]
        } for h in history]

        # extract information from history record
        num_gen_token_lst = {role: [] for role in agent_roles}
        stop_reason_lst = {role: [] for role in agent_roles}
        for h in history:
            _num_gen_tokens = {role: [] for role in agent_roles}
            _stop_reasons = {role: [] for role in agent_roles}
            for m in h:
                _num_gen_tokens[m['role']].append(m['num_gen_tokens'])
                _stop_reasons[m['role']].append(m['stop_reason'])
            for role in agent_roles:
                num_gen_token_lst[role].append(_num_gen_tokens[role])
                stop_reason_lst[role].append(_stop_reasons[role])

        tensor_dict = self._build_tensor_dict(
            last_round_responses,
            conversation_history, 
            tokenizer,
            num_gen_token_lst,
            stop_reason_lst,
            max_num_turns,
            finish_reason
        )

        # Prepare return results
        final_output = self._prepare_final_output(
            tensor_dict=tensor_dict,
            latest_outputs=latest_outputs,
            history=history,
            finish_reason=finish_reason,
            agent_roles=agent_roles,
            prompts=prompts,
            conversation_history=conversation_history,
        )

        if self.config.add_checking:
            ###################### TESTING ######################
            # 1. test lengths of history and conversation_history
            #  len(history[i]) == len(conversation_history[role][i]) * len(agent_roles)
            for i in range(len(history)):
                assert len(history[i]) == len(conversation_history[agent_roles[0]][i]), \
                    f"len(history[i]) = {len(history[i])} != len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])}"
                assert len(conversation_history[agent_roles[0]][i]) == len(conversation_history[agent_roles[1]][i]), \
                    f"len(conversation_history[agent_roles[0]][i]) = {len(conversation_history[agent_roles[0]][i])} != len(conversation_history[agent_roles[1]][i]) = {len(conversation_history[agent_roles[1]][i])}"

            # 2. check history role name order
            for i in range(len(history)):
                for j in range(len(history[i])):
                    assert history[i][j]['role'] == agent_roles[j % len(agent_roles)], \
                        f"history[i][j]['role'] = {history[i][j]['role']} != agent_roles[j % len(agent_roles)] = {agent_roles[j % len(agent_roles)]}"
                    
                # 2.1 check last round response
                for i_role, role in enumerate(agent_roles):
                    assert history[i][-len(agent_roles) + i_role]['role'] == role, \
                        f"history[i][-len(agent_roles) + i_role]['role'] = {history[i][-len(agent_roles) + i_role]['role']} != role = {role}"
                    assert history[i][-len(agent_roles) + i_role]['content'] == last_round_responses[i][role], \
                        f"history[i][-1]['content'] = {history[i][-1]['content']} != last_round_responses[i][role] = {last_round_responses[i][role]}"
                
            # 3. check conversation_history role name order
            for i_role, role in enumerate(conversation_history.keys()):
                for i in range(len(conversation_history[role])):
                    for j in range(len(conversation_history[role][i])):
                        if j == 0:
                            assert conversation_history[role][i][j]['role'] == "system", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'system'"
                        elif j % 2 == 1:
                            assert conversation_history[role][i][j]['role'] == "user", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'user'"
                        else:
                            assert conversation_history[role][i][j]['role'] == "assistant", \
                                f"conversation_history[role][i][j]['role'] = {conversation_history[role][i][j]['role']} != 'assistant'"
                            # check history string equals to conversation_string
                            assert conversation_history[role][i][j]['content'] == history[i][i_role + j - 2]['content'], \
                                f"'{[conversation_history[role][i][j]['content']]}' != '{[history[i][i_role + j - 2]['content']]}'"

            # 4. check input_ids
            for i_role, role in enumerate(agent_roles):
                role_tensor_dict = tensor_dict[role]
                for i in range(len(role_tensor_dict["input_ids"])):
                    input_ids = role_tensor_dict["input_ids"][i]
                    labels = role_tensor_dict["labels"][i]
                    attention_mask = role_tensor_dict["attention_mask"][i]
                    step_ids = role_tensor_dict["step_ids"][i]
                    stop_reasons = role_tensor_dict["stop_reasons"][i]
                    num_turn = final_output.non_tensor_batch["num_turns"][i]

                    query_response = tokenizer.decode(input_ids[attention_mask == 1].tolist())
                    raw_query_response = tokenizer.apply_chat_template(
                        conversation_history[role][i], 
                        add_generation_prompt=True, 
                        padding=True, 
                        truncation=False, 
                        max_length=None, 
                        tokenize=False, 
                    ) + last_round_responses[i][role]
                    
                    assert step_ids.max() == num_turn - 1 or stop_reasons[num_turn - 1] != 0, \
                        f"{step_ids.max()} != {num_turn - 1} or {stop_reasons[num_turn - 1]} != 0"

                    # FIXME: tokenizer has some issues on decode and encode unicode chars.

                    # assert query_response == raw_query_response, \
                    #     f"'{query_response}' != '{raw_query_response}'"
                    # for i_turn in range(num_turn):
                    #     turn_labels = labels[step_ids == i_turn]
                    #     if stop_reasons[i_turn] == 0:
                    #         assert turn_labels[-1] == tokenizer.eos_token_id
                    #         turn_labels = turn_labels[:-1] # drop eos
                    #     response = tokenizer.decode(turn_labels.tolist())
                    #     assert response == history[i][i_role + i_turn * len(agent_roles)]['content'], \
                    #         f"'{response}' != '{history[i][i_role + i_turn * len(agent_roles)]['content']}'"
        

        return final_output

    def _build_tensor_dict(self, 
                           last_round_responses: List[Dict[str, str]],
                           conversation_history: Dict[str, List[List[Dict[str, str]]]],
                           tokenizer: PreTrainedTokenizer,
                           num_gen_token_lst: Dict[str, List[List[int]]],
                           stop_reason_lst: Dict[str, List[List[Optional[str]]]],
                           max_num_turns: int,
                           finish_reason: List[Optional[str]]):
        # add last round output to make full conversation
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].append({
                    'role': 'assistant',
                    'content': last_round_responses[i_batch][role]
                })
        
        input_ids_lst = {role: [] for role in conversation_history.keys()}
        labels_lst = {role: [] for role in conversation_history.keys()}
        step_ids_lst = {role: [] for role in conversation_history.keys()}
        
        # build tensors for training
        for i_batch in range(len(last_round_responses)):
            for role in conversation_history.keys():
                # encode conversation into input_ids, labels, step_ids
                ####### DEPRECATED NOW ! #######
                # XXX(ziyu): to support reward model
                # input_ids shape is (seq_len + 1, ) if stop reason is 'stop' at
                #  the last turn.
                input_ids, labels, step_ids = encode_conversation(conversation_history[role][i_batch], 
                                                                  tokenizer, 
                                                                  num_gen_token_lst[role][i_batch], 
                                                                  stop_reason_lst[role][i_batch])
                input_ids_lst[role].append(input_ids)
                labels_lst[role].append(labels)
                step_ids_lst[role].append(step_ids)
        
        # Apply padding to create tensors
        batch_size = len(last_round_responses)
        tensor_dict = {}
        finish_reason_array = [] 
        for fr in finish_reason:
            if fr == "reach_max_turn":
                finish_reason_array.append(1)
            elif fr == "completion_token_exceeded":
                finish_reason_array.append(2)
            elif fr == "stop_when_truncated":
                finish_reason_array.append(3)
            elif fr is None:
                finish_reason_array.append(0)
            else:
                raise ValueError(f"Unknown finish reason: {fr}")
        
        
        for role in conversation_history.keys():
            # Find max length for padding
            max_length = max([len(ids) for ids in input_ids_lst[role]])
            if max_length > self.config.response_length + self.config.prompt_length:
                print(f"role: {role}, max_length={max_length} > {self.config.response_length + self.config.prompt_length}")
                # raise RuntimeError(f"max_length={max_length} > {self.config.response_length + self.config.prompt_length}")

            # Use max length for padding and gathering
            max_length = self.config.response_length + self.config.prompt_length
            
            # Pad and convert to tensors
            padded_input_ids = torch.full((batch_size, max_length), 
                                          tokenizer.pad_token_id, 
                                          dtype=torch.long)
            padded_labels = torch.full((batch_size, max_length), 
                                      -100,  # IGNORE_INDEX
                                      dtype=torch.long)
            padded_step_ids = torch.full((batch_size, max_length), 
                                          -100,  # IGNORE_INDEX
                                          dtype=torch.long)
            attention_mask = torch.zeros((batch_size, max_length), 
                                       dtype=torch.long)
            
            # Fill in the actual values
            for i, (input_ids, labels, step_ids) in enumerate(zip(
                    input_ids_lst[role], labels_lst[role], step_ids_lst[role])):
                seq_len = min(len(input_ids), max_length)
                padded_input_ids[i, :seq_len] = torch.tensor(input_ids[:seq_len], dtype=torch.long)
                padded_labels[i, :seq_len] = torch.tensor(labels[:seq_len], dtype=torch.long)
                padded_step_ids[i, :seq_len] = torch.tensor(step_ids[:seq_len], dtype=torch.long)
                attention_mask[i, :seq_len] = 1
            
            # Compute position ids from attention mask
            position_ids = compute_position_id_with_mask(attention_mask)
            
            padded_num_gen_tokens = torch.full((batch_size, max_num_turns), 
                                              0,
                                              dtype=torch.long)
            for i, num_gen_tokens in enumerate(num_gen_token_lst[role]):
                padded_num_gen_tokens[i, :len(num_gen_tokens)] = torch.tensor(num_gen_tokens, dtype=torch.long)
            padded_stop_reasons = torch.full((batch_size, max_num_turns), 
                                            0,
                                            dtype=torch.bool)

            for i, stop_reasons in enumerate(stop_reason_lst[role]):
                stop_reason_array = np.array([0 if r == 'stop' else 1 for r in stop_reasons])
                padded_stop_reasons[i, :len(stop_reason_array)] = torch.tensor(stop_reason_array, 
                                                                               dtype=torch.bool)

            # Create a separate tensor dict for each role
            tensor_dict[role] = dict(
                {
                    "input_ids": padded_input_ids,
                    "labels": padded_labels,
                    "step_ids": padded_step_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_gen_tokens": padded_num_gen_tokens,
                    "stop_reasons": padded_stop_reasons,
                    "turn_finished": torch.tensor(finish_reason_array),
                },
            )

        # remove side effect
        for i_batch in range(len(last_round_responses)):
            for role in last_round_responses[i_batch]:
                conversation_history[role][i_batch].pop()
        
        return tensor_dict
    
    def _initialize_conversation_state(self, batch_size):
        """Initialize conversation state variables"""
        history = [[] for _ in range(batch_size)]
        finish_flags = np.zeros(batch_size, dtype=bool)
        finish_reason = [None for _ in range(batch_size)]
        return history, finish_flags, finish_reason

    def _run_multi_turn_conversation(
        self,
        data_proto: DataProto,
        tokenizer: PreTrainedTokenizer,
        max_num_turns: int,
        agent_roles: List[str],
        finish_flag: str,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        response_length: int,
        **kwargs,
    ):
        """Execute multi-turn dialogue generation"""
        problem_inputs = data_proto.non_tensor_batch
        batch_size = len(problem_inputs['index'])
        problem_inputs = _reshape_dict(problem_inputs, batch_size)
        assert len(finish_flags) == len(
            problem_inputs), f"{finish_flags.shape} != {len(problem_inputs)}"

        conversation_history = {
            role: [None for _ in range(batch_size)]
            for role in agent_roles
        }

        for i_turn in range(max_num_turns):
            # Get indices of unfinished samples
            unfinished_indices = np.where(~finish_flags)[0]
            print(
                f"turn {i_turn+1} of {max_num_turns}, {len(unfinished_indices)}/{batch_size} unfinished"
            )

            if len(unfinished_indices) == 0:
                break
            # Each role takes turns generating in every round
            for i_role, role in enumerate(agent_roles):
                print(f"role: {role}")
                # Prepare prompts for current role
                prompt_proto, chat_lst = self._prepare_role_prompts(
                    role=role,
                    unfinished_indices=unfinished_indices,
                    history=history,
                    problem_inputs=problem_inputs,
                    agent_roles=agent_roles,
                    tokenizer=tokenizer,
                )

                # check current state length
                non_trunc_input = tokenizer.apply_chat_template(
                    chat_lst,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=False,
                    max_length=None,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True
                )
                # state length
                seq_lens = non_trunc_input["attention_mask"].sum(dim=1).tolist()
                # if state length is larger than prompt length, the trajectory is terminated 
                if not all([l <= self.config.prompt_length for l in seq_lens]):
                    # drop the terminated trajectories
                    new_seq_lens = []
                    new_unfinished_indices = []
                    new_prompt_protos = []
                    new_chat_lst = []
                    for i, idx in enumerate(unfinished_indices):
                        if seq_lens[i] <= self.config.prompt_length:
                            new_unfinished_indices.append(idx)
                            new_prompt_protos.append(prompt_proto[i])
                            new_seq_lens.append(seq_lens[i])
                            new_chat_lst.append(chat_lst[i])
                        else:
                            # set finish flag and finish reason 
                            finish_flags[idx] = True
                            finish_reason[idx] = "completion_token_exceeded"
                            print(f'idx={idx}, completion_token_exceeded')
                            # if the next gen is for reasoning agent, we need to add a dummy response in history
                            if role == agent_roles[1]:
                                history[idx].append(
                                    {"role": agent_roles[1], "content": "", 
                                    "num_gen_tokens": 0, 
                                    "stop_reason": "completion_token_exceeded"}
                                )
                                # update conversation history for reasoning agent
                                conversation_history[agent_roles[1]][idx] = chat_lst[i]
                            else:
                                if i_turn == 0:
                                    raise RuntimeError(f"1st round prompt larger than prompt length: {seq_lens[i]} > {self.config.prompt_length}")

                    # update unfinished_indices
                    unfinished_indices = new_unfinished_indices
                    if len(unfinished_indices) == 0:
                        break

                    # collate prompt needed to generate this round
                    prompt_proto = data_proto_collate_fn(new_prompt_protos)
                    seq_lens = new_seq_lens
                    chat_lst = new_chat_lst
                
                prompt_proto.meta_info.update(data_proto.meta_info)
                for i, chat in enumerate(chat_lst):
                    idx = unfinished_indices[i]
                    conversation_history[role][idx] = chat

                # Generate responses for current role
                current_outputs, num_gen_tokens, stop_reasons, resp_lens = self._generate_role_responses(
                    prompt_proto, tokenizer, response_length, **kwargs)
                
                # XXX(ziyu): remove finish flag in output for reasoning agent here
                #  consider move to a post-processing function
                if role == agent_roles[1] and finish_flag:
                    current_outputs = [
                        output.replace(finish_flag, "").rstrip() for output in current_outputs
                    ]
                

                # XXX(ziyu): side effect on `history`
                self._update_history_and_check_finish(
                    role=role,
                    current_outputs=current_outputs,
                    unfinished_indices=unfinished_indices,
                    history=history,
                    finish_flags=finish_flags,
                    finish_reason=finish_reason,
                    finish_flag=finish_flag,
                    agent_roles=agent_roles,
                    num_gen_tokens=num_gen_tokens,
                    stop_reasons=stop_reasons,
                    problem_inputs=problem_inputs,
                    conversation_history=conversation_history,
                    tokenizer=tokenizer,
                )
                unfinished_indices = np.where(~finish_flags)[0]
                if len(unfinished_indices) == 0:
                    break
            
        # use the last output of each agent as latest output response
        latest_outputs = [h[-1]['content'] for h in history]

        return latest_outputs, conversation_history

    def _prepare_role_prompts(
        self,
        role: str,
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        problem_inputs: List[Dict[str, str]],
        agent_roles: List[str],
        tokenizer,
    ) -> Tuple[DataProto, List[List[Dict[str, str]]]]:
        """Prepare prompts for a specific role"""

        # Prepare history and questions for currently unfinished samples
        current_history = [history[idx] for idx in unfinished_indices]
        current_problem_inputs = [problem_inputs[idx] for idx in unfinished_indices]

        # Build chat list
        chat_lst = self._build_chat_list_for_role(
            role,
            current_history,
            current_problem_inputs,
            agent_roles,
        )

        # Apply chat template and encode
        inputs = self._apply_chat_template(chat_lst, tokenizer)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        return data, chat_lst

    def _build_chat_list_for_role(
        self,
        role: str,
        history_list: List[List[Dict[str, str]]],
        problem_inputs: List[Dict[str, Any]],
        agent_roles: List[str],
    ):
        """Build chat list for a specific role"""

        chat_lst = [[] for _ in range(len(history_list))]

        for i, (hist, question_input) in enumerate(zip(history_list, problem_inputs)):
            input_template = question_input["input_template"]
            prompt_vars = dict(question_input)
            # Add system prompt when explicitly provided for this role.
            system_prompt = input_template.get("system_prompt", {}).get(role)
            if system_prompt:
                chat_lst[i].append({"role": "system", "content": system_prompt})
            if role == agent_roles[0]:  # attacker
                user_prompt = _fill_format_str(input_template['first_user_template'][role], prompt_vars)
                chat_lst[i].append({"role": "user", "content": user_prompt})
                for j in range(len(hist)):
                    if j % 2 == 0:
                        chat_lst[i].append({"role": "assistant", "content": hist[j]["content"]})
                    else:
                        chat_lst[i].append({"role": "user", "content": hist[j]["content"]})
            else:  # defender
                last_attacker_msg = next((m for m in reversed(hist) if m.get("role") == agent_roles[0]), None)
                attacker_visible = None
                if last_attacker_msg is not None:
                    attacker_visible = last_attacker_msg.get("content", "")
                if attacker_visible is None:
                    attacker_visible = prompt_vars.get("attacker_template") or prompt_vars.get("seed_prompt") or prompt_vars.get("question", "")
                attacker_answer = (extract_answer(attacker_visible) or attacker_visible).strip()
                attacker_think = (extract_think(attacker_visible) or "").strip()
                prompt_vars["attacker_prompt"] = attacker_answer
                prompt_vars["attacker_think"] = attacker_think
                user_prompt = _fill_format_str(input_template['first_user_template'][role], prompt_vars)
                chat_lst[i].append({"role": "user", "content": user_prompt})
                for j in range(1, len(hist)):
                    if (j + 1) % 2 == 0:
                        chat_lst[i].append({"role": "assistant", "content": hist[j]["content"]})
                    else:
                        chat_lst[i].append({"role": "user", "content": hist[j]["content"]})

        return chat_lst

    def _apply_chat_template(self, chat_lst: List[List[Dict[str, str]]],
                             tokenizer):
        """Apply chat template and encode"""
        return tokenizer.apply_chat_template(
            chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def _generate_role_responses(self, prompt_proto: DataProto, tokenizer,
                                 response_length: int, **kwargs):
        """Generate responses for the current role"""
        output = self.generate_sequences(prompt_proto, **kwargs)
        resp_lens = output.batch['attention_mask'][:, -response_length:].sum(dim=1).tolist()
        vllm_output_text = output.non_tensor_batch['text'].tolist()
        # output_text = tokenizer.batch_decode(
        #     output.batch["input_ids"][:, -response_length:],
        #     skip_special_tokens=False,
        # )

        # # Remove padding and EOS tokens from the output in one pass
        # pad_token = tokenizer.pad_token
        # eos_token = tokenizer.eos_token
        # output_text_clean = [
        #     text.replace(pad_token, "").replace(eos_token, "")
        #     for text in output_text
        # ]

        # for i, (decode_txt, vllm_txt) in enumerate(zip(output_text_clean, vllm_output_text)):
        #     if decode_txt != vllm_txt:
        #         print(f"i={i}, decode_txt={decode_txt}, vllm_txt={vllm_txt}")
        
        num_gen_tokens = output.non_tensor_batch['gen_response_lengths'].tolist()
        stop_reasons = output.non_tensor_batch['stop_reasons'].tolist()

        # return output_text_clean, num_gen_tokens, stop_reasons, resp_lens
        return vllm_output_text, num_gen_tokens, stop_reasons, resp_lens

    def _update_history_and_check_finish(
        self,
        role: str,
        current_outputs: List[str],
        unfinished_indices: np.ndarray,
        history: List[List[Dict[str, str]]],
        finish_flags: np.ndarray,
        finish_reason: List[Optional[str]],
        finish_flag: str,
        agent_roles: List[str],
        num_gen_tokens: List[int],
        stop_reasons: List[Optional[str]],
        problem_inputs: List[Dict[str, Any]],
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
        tokenizer: PreTrainedTokenizer,
    ):
        """Update conversation history and check completion flags"""
        # Update history
        assert len(current_outputs) == len(unfinished_indices), \
            f'{len(current_outputs)} != {len(unfinished_indices)}'
        for i, idx in enumerate(unfinished_indices):
            raw_output = current_outputs[i]
            history_entry = {
                "role": role,
                "content": raw_output,
                "num_gen_tokens": num_gen_tokens[i],
                "stop_reason": stop_reasons[i],
                "parsed_answer": (extract_answer(raw_output) or raw_output).strip(),
            }
            if role == agent_roles[0]:
                history_entry["parsed_think"] = (extract_think(raw_output) or "").strip()
            history[idx].append(history_entry)

        # Update finish flags
        # Check completion flags
        if role == agent_roles[1]:
            for i, idx in enumerate(unfinished_indices):
                last_output = history[idx][-2]
                assert last_output["role"] == agent_roles[0]
                response = last_output['content']
                if finish_flag and finish_flag in response:
                    finish_flags[idx] = True
                    finish_reason[idx] = None
        
        if self.config.stop_when_truncated:
            for i, stop_reason in enumerate(stop_reasons):
                # if stop_reason == "length" and not finish_flags[unfinished_indices[i]]:
                # XXX: even if stop by finish_flag, if current output is truncated, we need
                #  mark this trajectory as terminated
                if stop_reason == "length":
                    idx = unfinished_indices[i]
                    print(f'idx={idx}, stop_when_truncated')
                    finish_flags[idx] = True
                    finish_reason[idx] = "stop_when_truncated"
                    if role == agent_roles[0]:
                        # update conversation for reasoning agent
                        _, new_conversation = self._prepare_role_prompts(
                            role=agent_roles[1],
                            unfinished_indices=[idx],
                            history=history,
                            problem_inputs=problem_inputs,
                            agent_roles=agent_roles,
                            tokenizer=tokenizer,
                        )
                        conversation_history[agent_roles[1]][idx] = new_conversation[0]

                        # add dummy history of reasoning agent
                        history[idx].append(
                            {"role": agent_roles[1], "content": "", 
                             "num_gen_tokens": 0, 
                             "stop_reason": "stop_when_truncated"}
                        )

    def _mark_unfinished_as_max_turns(self, finish_flags: np.ndarray,
                                      finish_reason: List[Optional[str]]):
        """Mark unfinished samples as reaching maximum turns"""
        for i in range(len(finish_flags)):
            if not finish_flags[i]:
                finish_reason[i] = "reach_max_turn"

    def _prepare_final_output(
        self,
        tensor_dict: Dict[str, Dict[str, torch.Tensor]],
        latest_outputs: List[str],
        history: List[List[Dict[str, str]]],
        finish_reason: List[Optional[str]],
        agent_roles: List[str],
        prompts: DataProto,
        conversation_history: Dict[str, List[List[Dict[str, str]]]],
    ):
        """Prepare final output"""

        non_tensor_batch = prompts.non_tensor_batch
        non_tensor_batch["finish_reason"] = finish_reason
        non_tensor_batch["num_turns"] = [
            len(h) // len(agent_roles) for h in history
        ]
        non_tensor_batch["response"] = latest_outputs

        padded_history = _pad_history(history, 2 * self.config.max_num_turns)
        padded_conversation_history = {
            role:
            _pad_history(conversation_history[role],
                         2 * self.config.max_num_turns)
            for role in agent_roles
        }

        non_tensor_batch["history"] = padded_history
        for role in agent_roles:
            non_tensor_batch[
                f"{role}_conversation_history"] = padded_conversation_history[
                    role]

        flat_tensor_dict = {}
        for role in tensor_dict.keys():
            for key in tensor_dict[role].keys():
                flat_tensor_dict[f"{role}_{key}"] = tensor_dict[role][key]

        return DataProto.from_dict(
            tensors=flat_tensor_dict,
            non_tensors=non_tensor_batch,
            meta_info=prompts.meta_info,
        )


def _pad_history(input_historys: List[List[Dict[str, str]]],
                 max_length: int,
                 pad_value={
                     "role": "padding",
                     "content": "<PAD>"
                 }):
    padded_history = []
    for history in input_historys:
        current_length = len(history)
        pad_length = max_length - current_length
        assert pad_length >= 0, f"current_length: {current_length}, max_length: {max_length}"
        padded_history.append(history + [pad_value] * pad_length)
    return padded_history

def encode_conversation(conversation: List[Dict[str, str]], 
                        tokenizer: PreTrainedTokenizer, 
                        num_gen_tokens: List[int], 
                        stop_reasons: List[Optional[str]]):
    IGNORE_INDEX = -100
    labels = [] 
    step_ids = []
    cur_len = 0
    cur_hist = []
    i_step = 0
    for i, msg in enumerate(conversation):
        if msg['role'] in ['system', 'user']:
            pass
        elif msg['role'] == 'assistant':
            # query string
            query = tokenizer.apply_chat_template(
                cur_hist,
                add_generation_prompt=True,
                tokenize=False
            )
            # response string
            response = msg['content']
            query_ids = tokenizer.encode(query, add_special_tokens=True)
            query_response_ids = tokenizer.encode(query + response, add_special_tokens=True)
            response_ids = query_response_ids[len(query_ids):]
            input_ids = query_response_ids

            ################################################################
            # input_ids: 
            # | this | is | a | test | <im_end> | <im_start> | <assistant> | this | is | a | response | <im_end> |
            # query_ids:
            # | this | is | a | test | <im_end> | <im_start> | <assistant> |
            # response_ids:
            # | this | is | a | response | <im_end> |
            # step_ids:
            # |IGNORE| IG |IG | IG   | IG       | IG         | i_step      |i_step| ... |i_step| IGNORE |
            # labels:
            # |IGNORE| IG |IG | IG   | IG       | IG         | this | is   | a | response   | <im_end> | IGNORE
            #################################################################
            step_ids.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))
            labels.extend([IGNORE_INDEX] * (len(query_ids) - cur_len - 1))

            stop_reason = stop_reasons[i_step]
            # if stop normally, add eos token
            if stop_reason == "stop":
                labels.extend(response_ids + [tokenizer.eos_token_id])
                step_ids.extend([i_step] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = len(response_ids) + 1
            # if truncated, do not add eos token as label
            elif stop_reason == "length":
                # print("# STOP REASON:", stop_reasons[i_step])
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([i_step] * len(response_ids) + [IGNORE_INDEX])
                num_gen_tokens[i_step] = len(response_ids)
            elif stop_reason in ['stop_when_truncated', 'completion_token_exceeded']:
                # special case for dummy response
                # XXX: in this case, response == ""
                assert response == ""
                labels.extend(response_ids + [IGNORE_INDEX])
                step_ids.extend([IGNORE_INDEX] * (len(response_ids) + 1))
                num_gen_tokens[i_step] = 0
                break

            i_step += 1
            cur_len = len(query_response_ids)
        else:
            raise ValueError(f"Unknown message role: {msg['role']}")
        cur_hist.append(msg)

    assert len(input_ids) == len(labels), f"{len(input_ids)} != {len(labels)}"
    return input_ids, labels, step_ids
