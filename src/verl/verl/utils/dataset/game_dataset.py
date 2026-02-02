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

from omegaconf import ListConfig
import os
from typing import Dict, List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from functools import lru_cache
from string import Formatter

@lru_cache(maxsize=8)
def _add_prompts_for_data_source_and_turn(data_source: str, max_num_turns: int = 1):
    if data_source == "game":
        if max_num_turns == 1:
            from prompt.game.single_turn import (
                ATTACKER_SYSTEM_PROMPT,
                ATTACKER_FIRST_USER_PROMPT,
                DEFENDER_SYSTEM_PROMPT,
                DEFENDER_FIRST_USER_PROMPT,
            )
        elif max_num_turns > 1:
            from prompt.game.multi_turn import (
                ATTACKER_SYSTEM_PROMPT,
                ATTACKER_FIRST_USER_PROMPT,
                DEFENDER_SYSTEM_PROMPT,
                DEFENDER_FIRST_USER_PROMPT,
            )
        else:
            raise ValueError(f"{max_num_turns=} should be greater than or equal to 1")

    return {
        "system_prompt": {
            "attacker": ATTACKER_SYSTEM_PROMPT,
            "defender": DEFENDER_SYSTEM_PROMPT,
        },
        "first_user_template": {
            "attacker": ATTACKER_FIRST_USER_PROMPT,
            "defender": DEFENDER_FIRST_USER_PROMPT,
        },
    }

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class GameDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        prompt_key="question",
        max_num_turns=1,
    ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.prompt_key = prompt_key
        self.max_num_turns = max_num_turns

        # Whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # Read parquet files
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Return the prompt data directly without tokenization
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        # Get and pop the prompt from the row dictionary
        # question = row_dict.pop(self.prompt_key)

        # Store the raw chat data
        # row_dict['question'] = question

        # Retain the raw prompt template for downstream attacker/defender construction.
        prompt_messages = row_dict.get("prompt", None)
        if prompt_messages:
            row_dict["attacker_prompt_messages"] = prompt_messages
            first_message = prompt_messages[0]
            if isinstance(first_message, dict) and first_message.get("role") == "user":
                row_dict["attacker_template"] = first_message.get("content", "")

        # Cache the vanilla seed prompt so defender logic can reference it later if needed.
        seed_prompt = row_dict.get("extra_info", {}).get("raw_prompt")
        if seed_prompt is None:
            seed_prompt = row_dict.get(self.prompt_key, "")
        row_dict["seed_prompt"] = seed_prompt

        # Add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        data_source = row_dict["data_source"]
        row_dict["input_template"] = _add_prompts_for_data_source_and_turn(
            data_source, self.max_num_turns
        )
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()


if __name__ == "__main__":
    dataset = GameDataset(
        parquet_files=["data/MATH/train.parquet"], prompt_key="question"
    )
    print(len(dataset))
    print(dataset[0])
