import torch
import unittest
from unittest import mock

from algorithms.contrastive import ContrastiveLoss
from parameter.Parameter import Parameter


class DummyEp:
    def to(self, *args, **kwargs):
        return self

    def copy_weight_from(self, *args, **kwargs):
        return None


class RepresentationLearningConfigTest(unittest.TestCase):
    def test_contrastive_optimizer_tracks_weight_after_to(self):
        loss = ContrastiveLoss(c_dim=4, max_env_len=10, ep=DummyEp())

        loss.to(dtype=torch.float64)

        optimizer_param = loss.w_optim.param_groups[0]["params"][0]
        self.assertIs(optimizer_param, loss.W)
        self.assertEqual(optimizer_param.dtype, torch.float64)

    def test_wmcl_is_disabled_by_default_so_rmdm_remains_active(self):
        with mock.patch("sys.argv", ["test"]):
            parameter = Parameter()

        self.assertIs(parameter.use_rmdm, True)
        self.assertIs(parameter.use_wmcl, False)


if __name__ == "__main__":
    unittest.main()
