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
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'DigitalLearningGmbH/MATH-lighteval'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            prompt = question + ' ' + instruction_following

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": 'ReMA-math',
                "prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                },
                'question': question,
                'groundtruth': answer,
                'subset': 'MATH'
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # filter w.r.t. level, filter level 1 and 2
    def filter_by_level(example):
        level = example['level']
        # Keep only levels 1 and 2
        return level not in ['Level 1', 'Level 2']
    
    filtered_train_dataset = train_dataset.filter(filter_by_level)
    print(f"After filtering - Train: {len(filtered_train_dataset)}")
    

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    # filtered_train_dataset.to_parquet(os.path.join(local_dir, 'train_level3_5.parquet'))

    # # add first 16 rows of filtered_train_dataset as a train_minimal_dataset
    # train_minimal_dataset = filtered_train_dataset.select(range(16))
    # train_minimal_dataset.to_parquet(os.path.join(local_dir, 'train_minimal.parquet'))

    # # add first 16 level 5 problems as a train_level5_minimal_dataset
    # train_level5_minimal_dataset = train_dataset.filter(lambda x: x['level'] == 'Level 5').select(range(16))
    # train_level5_minimal_dataset.to_parquet(os.path.join(local_dir, 'train_level5_minimal.parquet'))

    # for each 'type', random sample 16 items from filtered_train_dataset
    problem_types = set(filtered_train_dataset['type'])
    print(f"Found {len(problem_types)} problem types")
    
    sampled_by_type = []
    for problem_type in problem_types:
        type_examples = filtered_train_dataset.filter(lambda x: x['type'] == problem_type)
        if len(type_examples) > 16:
            indices = random.sample(range(len(type_examples)), 19)
            sampled_examples = type_examples.select(indices)
        else:
            sampled_examples = type_examples
        sampled_by_type.append(sampled_examples)
    
    # Combine all sampled examples into one dataset
    train_type_samples = datasets.concatenate_datasets(sampled_by_type)
    print(f"Sample dataset by type size: {len(train_type_samples)}")
    
    # # Save the sampled dataset
    # os.makedirs(os.path.expanduser(local_dir), exist_ok=True)
    train_type_samples.to_parquet(os.path.join(local_dir, 'train_minimal_all_types.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
