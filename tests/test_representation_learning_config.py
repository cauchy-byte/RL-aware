import torch
import unittest
from unittest import mock
from types import SimpleNamespace

from algorithms.contrastive import ContrastiveLoss
from agent.Agent import EnvWorker
from envs.env_wrapper import create_env_decoration
from parameter.Parameter import Parameter
from tests.test_acda_observation_delay_env import DummyEnv


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

    def test_delay_process_defaults_to_legacy_and_accepts_acda_choices(self):
        with mock.patch("sys.argv", ["test"]):
            parameter = Parameter()
        self.assertEqual("legacy", parameter.delay_process)

        for name in ("ge1_23", "ge4_32", "mm1"):
            with mock.patch("sys.argv", ["test", "--delay_process", name]):
                parsed = Parameter()
            self.assertEqual(name, parsed.delay_process)

    def test_acda_training_controls_default_to_backward_compatible_values(self):
        with mock.patch("sys.argv", ["test"]):
            parameter = Parameter()

        self.assertEqual("random", parameter.delay_task_strategy)
        self.assertEqual(1, parameter.training_worker_num)

        with mock.patch(
            "sys.argv",
            ["test", "--delay_task_strategy", "acda_cycle",
             "--training_worker_num", "3"],
        ):
            parsed = Parameter()
        self.assertEqual("acda_cycle", parsed.delay_task_strategy)
        self.assertEqual(3, parsed.training_worker_num)

    def test_acda_wrapper_creates_process_tasks(self):
        parameter = SimpleNamespace(
            use_delay=True,
            delay_process="ge4_32",
            initial_delay_type="gamma",
            max_delay_range_min=3,
            max_delay_range_max=10,
            nonstationary_delay=True,
            delay_task_num=1,
            delay_changing_period=100,
            delay_changing_interval=10,
            delay_type_changing_period=1000,
            random_delay_per_episode=True,
        )
        env = create_env_decoration(parameter)(DummyEnv())
        self.assertEqual([{"delay_process": "ge4_32"}], env._delay_tasks)
        self.assertIsNone(env.delay_tasks)

    def test_acda_task_skips_legacy_scheduler_even_when_nonstationary(self):
        class SpyEnv:
            _delay_tasks = [{"delay_process": "mm1"}]

            def __init__(self):
                self.calls = []

            def set_delay_task(self, task):
                self.calls.append(("set_delay_task", task))

            def set_nonstationary_delay_para(self, *args, **kwargs):
                self.calls.append(("set_nonstationary_delay_para", args, kwargs))

        worker = object.__new__(EnvWorker)
        worker.env = SpyEnv()
        worker.non_stationary = True
        worker.task_ind = -1

        worker.change_env_param(set_env_ind=0)

        self.assertEqual(0, worker.task_ind)
        self.assertEqual([("set_delay_task", {"delay_process": "mm1"})], worker.env.calls)


if __name__ == "__main__":
    unittest.main()
