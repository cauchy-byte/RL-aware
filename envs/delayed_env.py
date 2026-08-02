from collections import deque
from random import sample
import itertools

import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete

import numpy as np

from envs.delayed_distribution import DoubleGaussianDistribution, GammaDistribution, UniformDistribution


class ObsDelayWrapper(gym.Wrapper):

    def __init__(self, env, obs_delay="gamma", initial_action=None, skip_initial_actions=False):
        super().__init__(env)

        if obs_delay == "gamma":
            self.obs_delay_dis = GammaDistribution()
        elif obs_delay == "uniform":
            self.obs_delay_dis = UniformDistribution()
        elif obs_delay =="doublegaussian":
            self.obs_delay_dis = DoubleGaussianDistribution()

        self.wrapped_env = env

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_observations = deque(maxlen=self.obs_delay_dis.max_delay+1)
        self.arrival_times_observations = deque(maxlen=self.obs_delay_dis.max_delay+1)

        self.t = 0
        self.done_signal_sent = False
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.done_signal_sent = False
        first_observation, _ = super().reset(**kwargs)

        # fill up buffers
        self.t = - (self.obs_delay_dis.max_delay+1)  # this is <= -1
        while self.t < 0:
            self.send_observation((first_observation, 0., False, False, {}, 0))
            self.t += 1

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation, {}

    def step(self, action):

        true_obs = action
        if self.done_signal_sent:
            self.send_observation(self.past_observations[0])
        else:
            m, r, terminated, truncated, info = self.env.step(action)
            true_obs = m
            self.cum_rew_actor += r
            self.done_signal_sent = terminated or truncated
            self.send_observation((m, self.cum_rew_actor, terminated, truncated, info, 0))

        m, cum_rew_actor_delayed, terminated, truncated, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        self.t += 1

        return m, r, terminated, truncated, info

    def send_observation(self, obs):
        alpha = self.obs_delay_dis.dis_sample()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        m, r, terminated, truncated, info, _ = self.past_observations[alpha]
        return m, r, terminated, truncated, info
