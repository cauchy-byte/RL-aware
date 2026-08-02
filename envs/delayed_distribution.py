import numpy as np

class GammaDistribution:
    def __init__(self, max_delay=6):
        self.max_delay = min(max_delay, 6)  # 限制最大延迟为6
        # 原始概率分布
        self.full_probs = np.array([0.40438, 0.29753, 0.16418, 0.08053, 0.03703, 0.01635])
        self._update_probs()
    
    def _update_probs(self):
        """根据当前max_delay更新概率分布"""
        if self.max_delay < 6:
            # 截断并重新归一化概率
            self.probs = self.full_probs[:self.max_delay].copy()
            self.probs = self.probs / self.probs.sum()
            self.delays = np.arange(1, self.max_delay + 1)
        else:
            self.probs = self.full_probs
            self.delays = np.arange(1, 7)
    
    def set_max_delay(self, max_delay):
        """动态设置最大延迟"""
        self.max_delay = min(max_delay, 6)
        self._update_probs()

    def dis_sample(self):
        return np.random.choice(self.delays, p=self.probs)

    def dis_probability(self):
        result = np.zeros(7)
        result[self.delays] = self.probs
        return result

 
class UniformDistribution:
    def __init__(self, max_delay=9):
        self.max_delay = min(max_delay, 9)
        self.full_probs = np.array([0.11112, 0.11111, 0.11111, 0.11111, 0.11111,
                                    0.11111, 0.11111, 0.11111, 0.11111])
        self._update_probs()

    def _update_probs(self):
        self.probs = self.full_probs[:self.max_delay].copy()
        self.probs = self.probs / self.probs.sum()
        self.delays = np.arange(1, self.max_delay + 1)

    def set_max_delay(self, max_delay):
        self.max_delay = min(max_delay, 9)
        self._update_probs()

    def dis_sample(self):
        return np.random.choice(self.delays, p=self.probs)

    def dis_probability(self):
        result = np.zeros(10)
        result[self.delays] = self.probs
        return result
    

class DoubleGaussianDistribution:
    def __init__(self, max_delay=10):
        self.max_delay = min(max_delay, 10)
        self.full_probs = np.array([0.03982, 0.12356, 0.15381, 0.13144, 0.10588,
                                    0.13168, 0.15682, 0.11015, 0.04098, 0.00586])
        self._update_probs()

    def _update_probs(self):
        self.probs = self.full_probs[:self.max_delay].copy()
        self.probs = self.probs / self.probs.sum()
        self.delays = np.arange(1, self.max_delay + 1)

    def set_max_delay(self, max_delay):
        self.max_delay = min(max_delay, 10)
        self._update_probs()

    def dis_sample(self):
        return np.random.choice(self.delays, p=self.probs)

    def dis_probability(self):
        result = np.zeros(11)
        result[self.delays] = self.probs
        return result
