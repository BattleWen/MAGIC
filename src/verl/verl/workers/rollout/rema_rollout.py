# from abc import ABC
# from typing import Dict, List, Union, Optional, Tuple
# from omegaconf import DictConfig
# import torch.nn as nn
# from tensordict import TensorDict
# from transformers import PreTrainedTokenizer

# from verl import DataProto
# from verl.workers.rollout.base import BaseRollout
# from verl.workers.rollout.hf_rollout import HFRollout
# from verl.workers.rollout.vllm_rollout import vLLMRollout
# from verl.workers.rollout.sglang_rollout import SGLangRollout

# class REMARollout(BaseRollout):
#     def __init__(
#         self,
#         mta_model: nn.Module,
#         ra_model: Optional[nn.Module],
#         mta_tokenizer: Optional[PreTrainedTokenizer],
#         ra_tokenizer: Optional[PreTrainedTokenizer],
#         config,
#     ):
#         super().__init__()
#         self.mta_model = mta_model
#         self.ra_model = ra_model
#         self.config = config

#         self.mta_tokenizer = mta_tokenizer
#         self.ra_tokenizer = ra_tokenizer
    
#     def _create_rollout(
#         self,
#         rollout_type: str,
#         model: nn.Module,
#         tokenizer: PreTrainedTokenizer,
#         config,
#         model_hf_config,
#         **kwargs
#     ) -> BaseRollout:
#         if rollout_type == 'hf':
#             return HFRollout(model, config)
#         elif rollout_type == 'vllm':
#             return vLLMRollout(model, config, tokenizer, model_hf_config, **kwargs)
#         elif rollout_type == 'sglang':
#             return SGLangRollout(model, config, tokenizer, model_hf_config, **kwargs)
#         else:
#             raise ValueError(f"Invalid rollout type: {rollout_type}")

#     def generate_sequences(self, prompts: DataProto) -> DataProto:
#         raw_messages = prompts.non_tensor_batch['raw_messages']
#         questions = prompts.non_tensor_batch['questions']
        
    
#     def _generate_mta_response(self, prompts: DataProto) -> DataProto:
#         raise NotImplementedError
    
#     def _generate_ra_response(self, prompts: DataProto) -> DataProto:
#         raise NotImplementedError
