from .api_judge import APIJudgeModel
from .base_judge_model import LLMJudgeModel
from .huggingface_judge import HFCausalJudgeModel

# Local testing without CUDA/vLLM installation will not support VLLMJudgeModel
try:
    from .vllm_judge import VLLMJudgeModel
except Exception:
    # To avoid import errors
    VLLMJudgeModel = None  # type: ignore
    pass
    # print(f"Could not import VLLMJudgeModel. VLLMJudgeModel will not be supported: {e}")
