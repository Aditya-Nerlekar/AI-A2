"""Hand-written reference policies.

These are NOT the project's solution - they exist only as comparison points
for the learned DQN policy.  Every baseline sees exactly the same (noisy)
observation vector as the agent; none of them peeks at hidden state.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .config import ACTIVITIES, BATCH, DEFER, DELIVER, SUPPRESS
from .env import OBS_INDEX

Policy = Callable[[np.ndarray], int]

_IDLE = ACTIVITIES.index("idle")
_PHONE = ACTIVITIES.index("on_phone")


def _activity(obs: np.ndarray) -> int:
    i = OBS_INDEX["activity"]
    return int(np.argmax(obs[i:i + len(ACTIVITIES)]))


def _tick_of_day(obs: np.ndarray, ticks_per_day: int = 288) -> int:
    angle = np.arctan2(obs[OBS_INDEX["time_sin"]], obs[OBS_INDEX["time_cos"]])
    return int(round((angle % (2 * np.pi)) / (2 * np.pi) * ticks_per_day)) % ticks_per_day


class AlwaysDeliver:
    """Default phone behaviour: every notification alerts immediately."""
    name = "always_deliver"

    def __call__(self, obs: np.ndarray) -> int:
        return DELIVER


class RandomPolicy:
    """Uniformly random action (lower reference)."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(4))


class FixedScheduleDigest:
    """Deliver urgent items now; hold everything else until a fixed slot.

    Non-urgent items are deferred (re-presented every 15 min) and released
    when the clock enters the first ``window`` ticks of each ``period``-tick
    slot, i.e. a digest roughly every ``period * 5`` minutes.
    """
    name = "fixed_schedule"

    def __init__(self, period_ticks: int = 12, window_ticks: int = 3, urgent: float = 0.7):
        self.period, self.window, self.urgent = period_ticks, window_ticks, urgent

    def __call__(self, obs: np.ndarray) -> int:
        if obs[OBS_INDEX["urgency"]] >= self.urgent:
            return DELIVER
        return DELIVER if _tick_of_day(obs) % self.period < self.window else DEFER


class DoNotDisturb:
    """Classic DND: alert only if urgent or the user looks idle/on the phone;
    otherwise drop to the digest (or suppress pure promotions)."""
    name = "do_not_disturb"

    def __init__(self, urgent: float = 0.7):
        self.urgent = urgent

    def __call__(self, obs: np.ndarray) -> int:
        if obs[OBS_INDEX["urgency"]] >= self.urgent or _activity(obs) in (_IDLE, _PHONE):
            return DELIVER
        promo = OBS_INDEX["category"] + 6   # "promotion"
        return SUPPRESS if obs[promo] > 0.5 else BATCH


class AttentionUrgencyThreshold:
    """Score = urgency + attention estimate; deliver above ``hi``, batch in the
    middle band, suppress clearly low-value items (tuned by hand once)."""
    name = "threshold"

    def __init__(self, hi: float = 1.0, lo: float = 0.35, urgent: float = 0.7):
        self.hi, self.lo, self.urgent = hi, lo, urgent

    def __call__(self, obs: np.ndarray) -> int:
        u, a = obs[OBS_INDEX["urgency"]], obs[OBS_INDEX["attention_est"]]
        if u >= self.urgent:
            return DELIVER
        score = u + a
        if score >= self.hi:
            return DELIVER
        if score >= self.lo or obs[OBS_INDEX["importance"]] > 0.6:
            return BATCH
        return SUPPRESS


def all_baselines(seed: int = 0) -> Dict[str, Policy]:
    """Fresh instances of every baseline, keyed by name."""
    items = [AlwaysDeliver(), RandomPolicy(seed), FixedScheduleDigest(),
             DoNotDisturb(), AttentionUrgencyThreshold()]
    return {p.name: p for p in items}
