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

from functools import partial
from typing import Dict

from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException

def compute_score_fn(compute_score, params):
    data_source, response, ground_truth, extra_info = params
    return compute_score(data_source, response, ground_truth, extra_info)


def _rema_math_format_reward_fn(role, response_str):
    if 'boxed' in response_str:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def _rema_laaj_format_reward_fn(role, response_str):
    from verl.utils.reward_score.pairwise_laaj import extract_final_verdict
    ans = extract_final_verdict(response_str)
    if ans is not None:
        if role == 'meta_thinking':
            return -0.25
        elif role == 'reasoning':
            return 0.25
        else:
            raise ValueError(f"Unknown {role=}") 
    else: return 0.0

def compute_format_r(data_source, role, response_str):
    if data_source == "ReMA-math":
        return _rema_math_format_reward_fn(role, response_str)
    elif data_source == 'ReMA-laaj':
        return _rema_laaj_format_reward_fn(role, response_str)
    else:
        raise ValueError(f'Unknown {data_source=} for format reward.')

class ReMARewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto)-> Dict[str, torch.Tensor]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        batch_size = len(data)
        max_num_turns = data.meta_info['max_num_turns']

        
        agent_roles = data.meta_info['agent_roles']
        reward_tensor_map = {
            f'{role}_turn_level_reward': torch.zeros(batch_size, max_num_turns, dtype=torch.float32) for role in agent_roles
        }
        
        already_print_data_sources = {}

        params = [
            (data[i].non_tensor_batch['data_source'],
             data[i].non_tensor_batch['response'],
             data[i].non_tensor_batch['reward_model']['ground_truth'],
             data[i].non_tensor_batch.get('extra_info', None),
             )
            for i in range(len(data))
        ]

        scores = []
        with ProcessPool(max_workers=1) as pool:
            future = pool.map(partial(compute_score_fn, self.compute_score), params, timeout=10)
            iterator = future.result()
            with tqdm(total=len(data), desc="Computing scores") as pbar:
                while True:
                    try:
                        result = next(iterator)
                        scores.append(result)
                    except TimeoutError:
                        print('Time Out')
                        scores.append(0.0)
                    except TimeoutException:
                        print('Math verify internal timeout')
                        scores.append(0.0)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        raise e
                    pbar.update(1)
        
        assert len(scores) == len(data)
        accuracy = torch.tensor(scores, dtype=torch.float32) # bsz
        reward_tensor_map['acc'] = accuracy
        for i_bsz in range(len(data)):
            data_item = data[i_bsz]  # DataProtoItem
            response_str = data_item.non_tensor_batch['response']
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            # extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            score = scores[i_bsz]
            
            num_turns = data_item.non_tensor_batch['num_turns']
            
            for i_role, role in enumerate(agent_roles):
                turn_finished = data_item.batch[f'{role}_turn_finished'].item()
                if data_item.meta_info['mask_unfinished_reward']:
                    # if conversation is not finised normally, i.e. with ['FINISH']
                    #  the reward should be zero.
                    # `turn_finished` is 0 means finished normally.
                    score = score if turn_finished == 0 else 0.0

                if turn_finished == 0 and data_item.meta_info['use_format_reward']:
                    last_round_msg = data_item.non_tensor_batch['history'][i_role]
                    assert last_round_msg['role'] == role, role

                    format_r = compute_format_r(data_source, role, last_round_msg['content'])
                    score += format_r
                reward_tensor_map[f'{role}_turn_level_reward'][i_bsz, num_turns - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                prompt_str = data_item.non_tensor_batch['question']
                padded_history = data_item.non_tensor_batch['history']
                history = padded_history[:num_turns * 2]
                already_print_data_sources[data_source] += 1
                print("[question]", prompt_str)
                print("[ground_truth]", ground_truth)
                print("[answer]", response_str)
                print("[score]", score)
                print("[history]", history)

        # Return both reward tensors in a dictionary
        return reward_tensor_map
