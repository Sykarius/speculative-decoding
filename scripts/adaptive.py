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

        self.draft_tokens_evaluated = 0
        self.verify_steps_taken = 0

    @profile
    def update_gamma(self, accepted: int, target_logits: torch.Tensor, draft_logits: torch.Tensor, device: str) -> int:

        self.verify_steps_taken += 1

        if self.config.type == 'aimd':
            self.aimd(accepted)
        elif self.config.type == 'entropy':
            self.entropy_based()
        elif self.config.type == 'jsd':
            self.jensen_shannon_distance(target_logits, draft_logits)
        else:
            raise ValueError(f"Unsupported adaptive method: {self.strategy}")

        return self.gamma
    
    def aimd(self, accepted):
        if accepted == self.gamma:
            self.gamma = min(self.gamma + self.config.step_size, self.config.gamma_max)
        else:
            self.gamma = max(int(self.gamma * self.config.decrease_factor), self.config.gamma_min)
    
    def entropy_early_exit(self, logits: torch.Tensor) -> bool:
        probs = torch.softmax(logits, dim=-1).clamp(min=1e-10)
        raw_entropy = - torch.sum(probs * torch.log(probs)).item()
        if self.entropy is None:
            self.entropy = raw_entropy
        else:
            self.entropy = self.config.smoothing_factor * self.entropy + (1 - self.config.smoothing_factor) * raw_entropy

        self.draft_tokens_evaluated += 1
        if self.draft_tokens_evaluated <= self.config.warmup_steps:
            return False

        return self.entropy > self.config.high_entropy_threshold

    def entropy_based(self):
        if self.entropy is not None and self.verify_steps_taken > self.config.warmup_steps:
            if self.entropy < self.config.low_entropy_threshold:
                new_gamma = self.gamma + self.config.step_size
                self.gamma = min(new_gamma, self.config.gamma_max)
            elif self.entropy > self.config.high_entropy_threshold:
                new_gamma = int(self.gamma * self.config.decrease_factor)
                self.gamma = max(new_gamma, self.config.gamma_min)
    
    def jensen_shannon_distance(self, target_logits, draft_logits):
        # Remove bonus token from target_slice
        target_prob = F.softmax(target_logits[:, :-1, :], dim=-1)
        draft_prob = F.softmax(draft_logits, dim=-1)
        
        target_prob = target_prob.clamp(min=1e-10)
        draft_prob = draft_prob.clamp(min=1e-10)
        m = (0.5 * (target_prob + draft_prob)).clamp(min=1e-10)
        
        kl_target = torch.sum(target_prob * (torch.log(target_prob) - torch.log(m)))
        kl_draft = torch.sum(draft_prob * (torch.log(draft_prob) - torch.log(m)))
        js_divergence = 0.5 * (kl_target + kl_draft)
        js_distance = torch.sqrt(js_divergence).item()

        if self.js_distance is None:
            self.js_distance = js_distance
        else:
            self.js_distance = self.config.smoothing_factor * self.js_distance + (1 - self.config.smoothing_factor) * js_distance

        if self.verify_steps_taken > self.config.warmup_steps:
            if self.js_distance < self.config.low_jsd_threshold:
                new_gamma = self.gamma + self.config.step_size
                self.gamma = min(new_gamma, self.config.gamma_max)
            elif self.js_distance > self.config.high_jsd_threshold:
                new_gamma = int(self.gamma * self.config.decrease_factor)
                self.gamma = max(new_gamma, self.config.gamma_min)

        return js_distance