"""Simulated smartphone user with hidden activity and attention.

The user's activity follows a *time-inhomogeneous Markov chain with
persistence*: every 5-minute tick the user keeps the current activity with
probability ``stickiness``; otherwise a new activity is drawn from the
profile's hour-of-day distribution for the current day type.  This produces
realistic runs of focus, meetings, sleep etc. while keeping the daily rhythm.

Hidden attention availability a_t in [0, 1] is the activity's mean attention
plus Gaussian jitter.  The agent never sees a_t or the true activity directly:
``observe`` returns a noisy attention estimate and an activity label that is
wrong with a small probability (e.g. a phone-sensor misclassification).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import ACTIVITIES, ATTENTION_MEAN, EnvConfig

_IDX = {name: i for i, name in enumerate(ACTIVITIES)}
HourPlan = List[Dict[str, float]]  # 24 entries of activity -> weight


def _plan(blocks: List[Tuple[int, int, Dict[str, float]]]) -> HourPlan:
    """Expand (start_hour, end_hour, weights) blocks into a 24-hour plan."""
    plan: HourPlan = [dict() for _ in range(24)]
    for start, end, weights in blocks:
        hours = range(start, end) if start < end else list(range(start, 24)) + list(range(0, end))
        for h in hours:
            plan[h] = dict(weights)
    missing = [h for h in range(24) if not plan[h]]
    if missing:
        raise ValueError(f"hour plan has gaps at {missing}")
    return plan


_SLEEP = {"sleeping": 1.0}
_LIGHT_SLEEP = {"sleeping": 0.8, "on_phone": 0.15, "idle": 0.05}
_WAKE = {"idle": 0.5, "on_phone": 0.4, "sleeping": 0.1}
_COMMUTE = {"commuting": 0.7, "on_phone": 0.2, "idle": 0.1}
_WORK = {"focused": 0.6, "meeting": 0.25, "idle": 0.1, "on_phone": 0.05}
_BREAK = {"idle": 0.5, "on_phone": 0.4, "focused": 0.1}
_EVENING = {"idle": 0.45, "on_phone": 0.3, "focused": 0.25}
_LEISURE = {"idle": 0.5, "on_phone": 0.35, "focused": 0.1, "commuting": 0.05}

# Each profile: {"weekday": HourPlan, "weekend": HourPlan}
PROFILES: Dict[str, Dict[str, HourPlan]] = {
    # University student: classes/labs 9-17 on weekdays, assignments at night.
    "student": {
        "weekday": _plan([
            (0, 1, _LIGHT_SLEEP), (1, 7, _SLEEP), (7, 8, _WAKE), (8, 9, _COMMUTE),
            (9, 13, _WORK), (13, 14, _BREAK), (14, 17, _WORK), (17, 18, _COMMUTE),
            (18, 23, _EVENING), (23, 24, _LIGHT_SLEEP),
        ]),
        "weekend": _plan([
            (0, 2, _LIGHT_SLEEP), (2, 9, _SLEEP), (9, 10, _WAKE),
            (10, 14, _LEISURE), (14, 17, _EVENING), (17, 23, _LEISURE),
            (23, 24, _LIGHT_SLEEP),
        ]),
    },
    # Night-shift worker: works 21:00-06:00, sleeps through the daytime peak
    # of notification traffic.  Used only for the generalisation test.
    "night_shift": {
        "weekday": _plan([
            (21, 6, _WORK), (6, 7, _COMMUTE), (7, 9, _WAKE),
            (9, 16, _SLEEP), (16, 17, _WAKE), (17, 20, _LEISURE), (20, 21, _COMMUTE),
        ]),
        "weekend": _plan([
            (0, 4, _LEISURE), (4, 12, _SLEEP), (12, 13, _WAKE), (13, 24, _LEISURE),
        ]),
    },
}


class SimulatedUser:
    """Hidden user state machine for one simulated day."""

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator,
                 obs_rng: np.random.Generator | None = None):
        if cfg.profile not in PROFILES:
            raise KeyError(f"unknown profile {cfg.profile!r}; choose from {list(PROFILES)}")
        self.cfg = cfg
        self.rng = rng                      # drives the hidden dynamics only
        # separate stream for sensor noise so that the hidden day is identical
        # for every policy evaluated on the same seed
        self.obs_rng = obs_rng if obs_rng is not None else rng
        self.weekday = True
        self.activity = 0
        self.attention = 0.0
        self._tables = {
            kind: self._to_matrix(plan) for kind, plan in PROFILES[cfg.profile].items()
        }

    @staticmethod
    def _to_matrix(plan: HourPlan) -> np.ndarray:
        """24 x n_activities row-normalised probability table."""
        mat = np.zeros((24, len(ACTIVITIES)))
        for h, weights in enumerate(plan):
            for name, w in weights.items():
                mat[h, _IDX[name]] = w
            mat[h] /= mat[h].sum()
        return mat

    # ------------------------------------------------------------------
    def start_day(self, weekday: bool) -> None:
        self.weekday = weekday
        self.activity = int(self.rng.choice(len(ACTIVITIES), p=self._table()[0]))
        self._refresh_attention()

    def _table(self) -> np.ndarray:
        return self._tables["weekday" if self.weekday else "weekend"]

    def _refresh_attention(self) -> None:
        mean = ATTENTION_MEAN[ACTIVITIES[self.activity]]
        self.attention = float(np.clip(mean + self.rng.normal(0.0, self.cfg.attention_noise), 0.0, 1.0))

    def advance(self, tick: int) -> None:
        """Move the hidden state forward to ``tick`` (5-minute resolution)."""
        hour = (tick * self.cfg.tick_minutes // 60) % 24
        if self.rng.random() > self.cfg.activity_stickiness:
            self.activity = int(self.rng.choice(len(ACTIVITIES), p=self._table()[hour]))
        else:
            # A user cannot stay in an activity the profile forbids at this hour
            # (e.g. still "in a meeting" at 3 am): resample if probability is 0.
            if self._table()[hour, self.activity] == 0.0:
                self.activity = int(self.rng.choice(len(ACTIVITIES), p=self._table()[hour]))
        self._refresh_attention()

    def observe(self) -> Tuple[int, float]:
        """Noisy (activity_index, attention_estimate) visible to the agent."""
        act = self.activity
        if self.obs_rng.random() < self.cfg.obs_activity_flip:
            act = int(self.obs_rng.integers(len(ACTIVITIES)))
        est = float(np.clip(self.attention + self.obs_rng.normal(0.0, self.cfg.obs_attention_noise), 0.0, 1.0))
        return act, est

    @property
    def activity_name(self) -> str:
        return ACTIVITIES[self.activity]
