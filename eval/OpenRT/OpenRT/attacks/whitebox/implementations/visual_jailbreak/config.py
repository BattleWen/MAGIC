from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class VisualJailbreakConfig:
    """Configuration for Visual Jailbreak Attack"""
    
    # Attack parameters
    num_iter: int = 2000
    batch_size: int = 8
    alpha: float = 1/255  # Step size for adversarial noise
    epsilon: float = 128/255  # Maximum perturbation bound for constrained attack
    
    # Attack mode
    constrained: bool = True  # Whether to use constrained attack (with epsilon bound)
    
    # Target generation
    max_new_tokens: int = 1024
    temperature: float = 0.2
    do_sample: bool = True
    
    # Logging and visualization
    log_interval: int = 20  # Interval for plotting loss
    test_interval: int = 100  # Interval for testing generation
    save_temp_images: bool = True
    save_dir: str = "./results/visual_attack_results"
    
    # Normalization parameters (standard ImageNet normalization)
    mean: List[float] = None
    std: List[float] = None
    
    # Device
    device: str = "cuda"
    
    # Random seed for reproducibility
    seed: Optional[int] = None
    
    # Corpus-based optimization settings
    use_derogatory_corpus: bool = True  # Whether to use derogatory corpus as target outputs
    corpus_sample_size: Optional[int] = None  # Maximum number of corpus entries to use (None = use all)
    save_adv_images: bool = True # Whether to save adversarial images
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.48145466, 0.4578275, 0.40821073]
        if self.std is None:
            self.std = [0.26862954, 0.26130258, 0.27577711]

@dataclass
class VisualJailbreakResult:
    """Result of Visual Jailbreak Attack"""
    adversarial_image: Optional[torch.Tensor] = None
    loss_history: List[float] = None
    final_loss: float = float('inf')
    generation_outputs: List[str] = None
    success: bool = False
