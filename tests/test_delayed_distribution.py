import unittest

import numpy as np

from envs.delayed_distribution import (
    DoubleGaussianDistribution,
    GammaDistribution,
    UniformDistribution,
)


class DelayedDistributionTest(unittest.TestCase):
    def assert_respects_max_delay(self, distribution_cls, max_delay):
        np.random.seed(0)
        distribution = distribution_cls(max_delay=max_delay)

        samples = np.array([distribution.dis_sample() for _ in range(1000)])
        probs = distribution.dis_probability()

        self.assertLessEqual(samples.max(), max_delay)
        self.assertAlmostEqual(1.0, probs.sum(), places=6)
        self.assertTrue(np.all(probs[max_delay + 1:] == 0.0))

    def test_gamma_respects_configured_max_delay(self):
        self.assert_respects_max_delay(GammaDistribution, 3)

    def test_uniform_respects_configured_max_delay(self):
        self.assert_respects_max_delay(UniformDistribution, 3)

    def test_doublegaussian_respects_configured_max_delay(self):
        self.assert_respects_max_delay(DoubleGaussianDistribution, 3)


if __name__ == "__main__":
    unittest.main()
