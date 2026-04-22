from config import AdaptiveConfig
import torch
import torch.nn.functional as F
from metrics import profile

class AdaptiveController:

    def __init__(self, gamma: int, config: AdaptiveConfig):
        if config is None:
            raise ValueError("AdaptiveController requires a valid AdaptiveConfig.")
        
        self.gamma = gamma
        self.config = config
        self.strategy = config.strategy

        self.early_stop =  self.strategy == "entropy" or self.strategy == "jsd"
        
        self.entropy = None
        self.js_distance = None

    @profile
    def update_gamma(self, accepted: int, device: str) -> int:
        if self.config.type == 'aimd':
            return self.aimd(accepted)
        elif self.config.type == 'entropy':
            return self.gamma
        elif self.config.type == 'jsd':
            return self.gamma
        else:
            raise ValueError(f"Unsupported adaptive method: {self.config.type}")
    
    def aimd(self, accepted):
        if accepted == self.gamma:
            self.gamma = min(self.gamma + 1, self.gamma_range[1])
        else:
            self.gamma = max(self.gamma // 2, self.gamma_range[0])
        return self.gamma
    
    def entropy_early_exit(self, logits: torch.Tensor) -> bool:
        probs = torch.softmax(logits, dim=-1)
        raw_entropy = - torch.sum(probs * torch.log(probs)).item()
        if self.entropy is None:
            self.entropy = raw_entropy
        else:
            self.entropy = self.config.smoothing_factor * self.entropy + (1 - self.config.smoothing_factor) * raw_entropy
        return self.entropy > self.config.high_threshold
