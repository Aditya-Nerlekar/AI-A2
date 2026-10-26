"""Notification stream generator.

Arrivals follow a non-homogeneous Poisson process on the 5-minute grid:
N_t ~ Poisson(lambda_t) with lambda_t = daily_arrivals * w(hour) / sum(w)/12,
where w(hour) is a "world traffic" curve (quiet at night, peaks around the
morning, lunch and evening).  The curve depends on the clock, not on the user,
so a night-shift user sleeps through the busiest part of the day.

The whole day's stream is drawn up front from its own RNG, which makes the
arrivals identical for every policy evaluated with the same seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .config import CATEGORIES, CATEGORY_SPECS, EnvConfig

# Relative notification traffic by hour of day (00..23).
WORLD_TRAFFIC = np.array([
    0.15, 0.10, 0.05, 0.05, 0.05, 0.10, 0.30, 0.80,   # 00-07
    1.20, 1.30, 1.20, 1.10, 1.30, 1.40, 1.10, 1.00,   # 08-15
    1.10, 1.20, 1.40, 1.50, 1.60, 1.40, 1.00, 0.50,   # 16-23
])

_SHARES = np.array([CATEGORY_SPECS[c].share for c in CATEGORIES])
_SHARES = _SHARES / _SHARES.sum()


@dataclass
class Notification:
    """One notification and its bookkeeping inside the simulator."""
    uid: int
    arrival: int            # tick of arrival
    category: int           # index into CATEGORIES
    urgency: float          # in [0, 1]
    relevance: float        # hidden task relevance in [0, 1]
    importance: float       # observable sender importance in [0, 1]
    salience: float
    deadline: int           # absolute tick after which it expires
    urgent: bool
    times_deferred: int = 0
    ready_at: int = field(default=0)   # tick when a deferred item comes back

    def age(self, tick: int) -> int:
        return tick - self.arrival


class NotificationStream:
    """Draws and serves the arrivals of one simulated day."""

    def __init__(self, cfg: EnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.by_tick: List[List[Notification]] = []
        self._uid = 0

    def rate(self, tick: int) -> float:
        hour = (tick * self.cfg.tick_minutes // 60) % 24
        ticks_per_hour = 60 // self.cfg.tick_minutes
        return self.cfg.daily_arrivals * WORLD_TRAFFIC[hour] / WORLD_TRAFFIC.sum() / ticks_per_hour

    def generate_day(self) -> None:
        self._uid = 0
        self.by_tick = []
        for t in range(self.cfg.ticks_per_day):
            n = int(self.rng.poisson(self.rate(t)))
            self.by_tick.append([self._make(t) for _ in range(n)])

    def _make(self, tick: int) -> Notification:
        cfg, rng = self.cfg, self.rng
        cat = int(rng.choice(len(CATEGORIES), p=_SHARES))
        spec = CATEGORY_SPECS[CATEGORIES[cat]]
        urgency = float(rng.uniform(*spec.urgency))
        ttl = spec.ttl_ticks
        importance = float(rng.beta(2.0, 2.0))
        # call-like urgent chat messages (missed calls, "are you there?")
        if CATEGORIES[cat] == "messaging" and rng.random() < cfg.urgent_message_prob:
            urgency = float(rng.uniform(0.8, 1.0))
            ttl = 3
            importance = max(importance, 0.7)
        relevance = spec.relevance * (0.5 + importance) + rng.normal(0.0, cfg.relevance_noise)
        relevance = float(np.clip(relevance, 0.0, 1.0))
        note = Notification(
            uid=self._uid, arrival=tick, category=cat, urgency=urgency,
            relevance=relevance, importance=importance, salience=spec.salience,
            deadline=tick + ttl, urgent=urgency >= cfg.urgent_threshold, ready_at=tick,
        )
        self._uid += 1
        return note

    def arrivals(self, tick: int) -> List[Notification]:
        return self.by_tick[tick] if 0 <= tick < len(self.by_tick) else []

    @property
    def total(self) -> int:
        return sum(len(x) for x in self.by_tick)
