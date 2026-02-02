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
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from math_verify.errors import TimeoutException
from tqdm import tqdm

def compute_score_fn(compute_score, params):
    data_source, response, ground_truth, extra_info = params
    return compute_score(data_source, response, ground_truth, extra_info)

class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score

    def verify(self, data):
        raise NotImplementedError("Verify is not implemented for NaiveRewardManager")
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

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        response_str_lst = []
        for i in range(len(data)):
            # decode response str
            data_item = data[i]  # DataProtoItem
            
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode the response and store it in non_tensor_batch
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str_lst.append(response_str)

        params = [
            (data[i].non_tensor_batch['data_source'],
             response_str_lst[i],
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

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response_ids = data_item.batch['responses']
            # valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = response_str_lst[i]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            # extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # score = self.compute_score(
            #     data_source=data_source,
            #     solution_str=response_str,
            #     ground_truth=ground_truth,
            #     extra_info=extra_info,
            # )
            score = scores[i]
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor
