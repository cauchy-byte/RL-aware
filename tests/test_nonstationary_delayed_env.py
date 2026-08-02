import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.nonstationary_delayed_env import NonstationaryDelayedEnv


class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 0.0, False, False, {}


class NonstationaryDelayedEnvTest(unittest.TestCase):
    def test_reset_keeps_explicit_delay_task_when_episode_randomization_enabled(self):
        env = NonstationaryDelayedEnv(
            DummyEnv(),
            max_delay_range=(9, 9),
            random_delay_per_episode=True,
        )

        env.set_delay_task({"delay_type": "gamma", "max_delay": 3})
        env.reset()

        self.assertEqual("gamma", env.current_delay_type)
        self.assertEqual(3, env.current_max_delay)


if __name__ == "__main__":
    unittest.main()
