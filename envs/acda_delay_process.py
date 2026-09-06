"""ACDA-derived stateful processes used as observation-delay samplers.

The processes in this module are adapted from the delay-process semantics used
by a third-party ACDA reimplementation.  This module intentionally exposes
only the process interface needed by :class:`NonstationaryDelayedEnv`; it does
not implement ACDA's action-packet interaction layer.
"""

from typing import Protocol

import numpy as np


class ObservationDelayProcess(Protocol):
    name: str
    max_delay: int
    regime: str
    last_was_censored: bool

    def sample(self) -> int:
        ...

    def reset(self) -> None:
        ...

    def probability_vector(self, support_size: int = 33) -> np.ndarray:
        ...


def _validate_support_size(support_size: int, max_delay: int) -> None:
    if not isinstance(support_size, int) or support_size <= max_delay:
        raise ValueError(
            f"support_size must be an integer greater than max_delay={max_delay}, "
            f"got {support_size!r}"
        )


class _GilbertElliottProcess:
    def __init__(
        self,
        name: str,
        rng: np.random.Generator,
        good_delays: tuple[int, ...],
        good_probs: tuple[float, ...],
        bad_delays: tuple[int, ...],
        bad_probs: tuple[float, ...],
        bad_given_good: float,
        good_given_bad: float,
    ):
        self.name = name
        self.max_delay = max(max(good_delays), max(bad_delays))
        self._rng = rng
        self._good_delays = np.asarray(good_delays, dtype=np.int64)
        self._good_probs = np.asarray(good_probs, dtype=np.float64)
        self._bad_delays = np.asarray(bad_delays, dtype=np.int64)
        self._bad_probs = np.asarray(bad_probs, dtype=np.float64)
        self._bad_given_good = float(bad_given_good)
        self._good_given_bad = float(good_given_bad)
        self.reset()

    @property
    def regime(self) -> str:
        return self._regime

    @property
    def last_was_censored(self) -> bool:
        return False

    def reset(self) -> None:
        self._regime = "good"
        self._last_sampled_regime = "good"

    def sample(self) -> int:
        sampled_regime = self._regime
        if sampled_regime == "good":
            delay = self._rng.choice(self._good_delays, p=self._good_probs)
            transition_probability = self._bad_given_good
            next_regime = "bad" if float(self._rng.random()) < transition_probability else "good"
        else:
            delay = self._rng.choice(self._bad_delays, p=self._bad_probs)
            transition_probability = self._good_given_bad
            next_regime = "good" if float(self._rng.random()) < transition_probability else "bad"

        self._last_sampled_regime = sampled_regime
        self._regime = next_regime
        return int(delay)

    def probability_vector(self, support_size: int = 33) -> np.ndarray:
        _validate_support_size(support_size, self.max_delay)
        if self._last_sampled_regime == "good":
            delays, probabilities = self._good_delays, self._good_probs
        else:
            delays, probabilities = self._bad_delays, self._bad_probs
        result = np.zeros(support_size, dtype=np.float64)
        result[delays] = probabilities
        return result


class _MM1Process:
    name = "mm1"
    max_delay = 16
    regime = "queue"

    def __init__(
        self,
        rng: np.random.Generator,
        lambda_arrive: float = 0.33,
        lambda_service: float = 0.75,
    ):
        if lambda_arrive <= 0 or lambda_service <= 0:
            raise ValueError("lambda_arrive and lambda_service must be positive")
        if lambda_arrive >= lambda_service:
            raise ValueError("lambda_arrive must be smaller than lambda_service")
        self._rng = rng
        self.lambda_arrive = float(lambda_arrive)
        self.lambda_service = float(lambda_service)
        self._rate_gap = self.lambda_service - self.lambda_arrive
        self.reset()

    @property
    def last_was_censored(self) -> bool:
        return self._last_was_censored

    def reset(self) -> None:
        self._arrival_time = 0.0
        self._last_completion_time = 0.0
        self._last_was_censored = False

    def sample(self) -> int:
        self._arrival_time += float(self._rng.exponential(1.0 / self.lambda_arrive))
        service_time = float(self._rng.exponential(1.0 / self.lambda_service))
        completion_time = max(self._arrival_time, self._last_completion_time) + service_time
        self._last_completion_time = completion_time

        raw_delay = max(1, int(np.ceil(completion_time - self._arrival_time)))
        self._last_was_censored = raw_delay >= self.max_delay
        return min(raw_delay, self.max_delay)

    def probability_vector(self, support_size: int = 33) -> np.ndarray:
        _validate_support_size(support_size, self.max_delay)
        result = np.zeros(support_size, dtype=np.float64)
        rate = self._rate_gap
        for delay in range(1, self.max_delay):
            result[delay] = np.exp(-rate * (delay - 1)) - np.exp(-rate * delay)
        result[self.max_delay] = np.exp(-rate * (self.max_delay - 1))
        return result


def create_acda_observation_delay_process(
    name: str,
    rng: np.random.Generator,
) -> ObservationDelayProcess:
    """Create one canonical ACDA-derived observation-delay process."""
    if name == "ge1_23":
        return _GilbertElliottProcess(
            name=name,
            rng=rng,
            good_delays=(1, 2),
            good_probs=(15.0 / 16.0, 1.0 / 16.0),
            bad_delays=(22, 23, 24),
            bad_probs=(3.0 / 11.0, 5.0 / 11.0, 3.0 / 11.0),
            bad_given_good=1.0 / 125.0,
            good_given_bad=1.0 / 20.0,
        )
    if name == "ge4_32":
        return _GilbertElliottProcess(
            name=name,
            rng=rng,
            good_delays=(4,),
            good_probs=(1.0,),
            bad_delays=(32,),
            bad_probs=(1.0,),
            bad_given_good=1.0 / 250.0,
            good_given_bad=1.0 / 32.0,
        )
    if name == "mm1":
        return _MM1Process(rng=rng)
    raise ValueError(f"Unsupported ACDA observation delay process: {name!r}")

