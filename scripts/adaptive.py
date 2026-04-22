from config import AdaptiveConfig

class Adaptive:

    def __init__(self, gamma: int, config: AdaptiveConfig | None = None):
        self.gamma = gamma
        self.config = config
    
    def update_gamma(self, accepted: int) -> int:
        if self.config is None:
            return self.gamma
        elif self.config.name == 'aimd':
            if accepted == self.gamma:
                self.gamma = min(self.gamma + 1, self.config.gamma_max)
            else:
                self.gamma = max(self.gamma // 2, self.config.gamma_min)
            return self.gamma
        else:
            raise ValueError(f"Unsupported adaptive method: {self.config.name}")
