import copy
from dataclasses import dataclass, field
from typing import Optional, Union, List

@dataclass
class GCGConfig:
    """GCG attack configuration"""
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: Optional[int] = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: Optional[int] = None
    verbosity: str = "INFO"
    target_output: str = "I cannot help with that request."  # User-specified target output

@dataclass
class GCGResult:
    """GCG attack result"""
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    success: bool = False
    final_prompt: str = ""
    output_text: str = ""