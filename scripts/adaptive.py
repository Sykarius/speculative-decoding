from config import AdaptiveConfig
import torch
import torch.nn.functional as F
from metrics import profile
from common import compute_js_distance

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

        self.avg_da = None
        self.avg_dr = None
        self.threshold_v = None

        if self.strategy == "ada":
            self.avg_da = config.avg_da
            self.avg_dr = config.avg_dr
            self.threshold_v = (self.avg_da + self.avg_dr) / 2


        self.draft_tokens_evaluated = 0
        self.verify_steps_taken = 0

    @profile
    def update_gamma(self, accepted: int, target_logits: torch.Tensor, draft_logits: torch.Tensor, device: str) -> int:

        self.verify_steps_taken += 1

        if self.strategy == 'aimd':
            self.aimd(accepted)
        elif self.strategy == 'entropy':
            self.entropy_based()
        elif self.strategy == 'jsd':
            self.jensen_shannon_distance(target_logits, draft_logits)
        elif self.strategy == 'ada':
            pass # Do nothing
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
        if self.entropy is not None and self.config.resize and self.verify_steps_taken > self.config.warmup_steps:
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
        
        js_distance_per_token = compute_js_distance(target_prob, draft_prob)
        js_distance = torch.mean(js_distance_per_token).item()

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

    def update_threshold(self, accepted_dists, rejected_dist):

        if self.strategy != "ada":
            raise ValueError(f"The adaptive strategy must be ada to update threshold")

        for d in accepted_dists:
            self.avg_da = self.config.smoothing_factor * self.avg_da + (1 - self.config.smoothing_factor) * d
        
        if rejected_dist is not None:
            self.avg_dr = self.config.smoothing_factor * self.avg_da + (1 - self.config.smoothing_factor) * rejected_dist

        self.threshold_v = (self.avg_da + self.avg_dr) / 2