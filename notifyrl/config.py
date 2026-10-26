"""Central configuration for NotifyRL.

Every tunable number in the project lives here so that experiments are
reproducible from a single place.  Values are *assumptions* chosen to give a
plausible, internally consistent synthetic world; they are NOT fitted to real
user data.  The rationale for each group is written next to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Discrete vocabularies
# ---------------------------------------------------------------------------
ACTIVITIES: Tuple[str, ...] = (
    "sleeping", "focused", "meeting", "commuting", "idle", "on_phone",
)
CATEGORIES: Tuple[str, ...] = (
    "messaging", "email", "social", "news", "calendar", "otp_delivery", "promotion",
)
ACTIONS: Tuple[str, ...] = ("deliver", "defer", "batch", "suppress")

DELIVER, DEFER, BATCH, SUPPRESS = 0, 1, 2, 3

# Mean hidden attention availability per activity (before noise).
# Sleeping users are essentially unreachable; a user already holding the phone
# is almost free to look at a notification.
ATTENTION_MEAN: Dict[str, float] = {
    "sleeping": 0.02, "focused": 0.25, "meeting": 0.10,
    "commuting": 0.45, "idle": 0.80, "on_phone": 0.95,
}


@dataclass(frozen=True)
class CategorySpec:
    """Generative parameters for one notification category.

    share      : fraction of all arrivals belonging to this category
    urgency    : (low, high) uniform range of the urgency score
    relevance  : mean task-relevance value in [0, 1]
    salience   : how disruptive the alert is (sound/vibration/banner size)
    ttl_ticks  : life time in 5-minute ticks; urgent items expire hard at this
                 deadline, others just lose value as they age
    """
    share: float
    urgency: Tuple[float, float]
    relevance: float
    salience: float
    ttl_ticks: int


CATEGORY_SPECS: Dict[str, CategorySpec] = {
    # chats from people: moderately relevant, sometimes call-like urgent
    "messaging":    CategorySpec(0.30, (0.20, 0.70), 0.60, 0.70, 36),
    "email":        CategorySpec(0.18, (0.10, 0.50), 0.45, 0.50, 96),
    "social":       CategorySpec(0.18, (0.00, 0.30), 0.25, 0.60, 48),
    "news":         CategorySpec(0.10, (0.00, 0.30), 0.20, 0.50, 48),
    # reminders and one-time passwords are useless after a short deadline
    "calendar":     CategorySpec(0.06, (0.75, 1.00), 0.85, 0.90, 3),
    "otp_delivery": CategorySpec(0.06, (0.80, 1.00), 0.90, 0.80, 2),
    "promotion":    CategorySpec(0.12, (0.00, 0.15), 0.08, 0.50, 72),
}


@dataclass
class RewardWeights:
    """Weights of the reward terms (see env.py docstring for the formula).

    w_value     1.0  : unit of benefit; relevance x engagement is in [0, 1].
    w_interrupt 1.0  : an alert at zero attention with full salience costs as
                       much as the best possible delivery gains, so the agent
                       must genuinely trade benefit against disruption.
    w_miss      3.0  : missing an OTP / meeting reminder is several times
                       worse than any single interruption.
    w_stale     0.02 : per tick, per unit urgency of held items; small enough
                       that holding a low-urgency item for an hour (12 ticks)
                       costs < 0.1, large enough to discourage endless deferral.
    w_fatigue   0.3  : per interruption above the hourly budget.
    w_digest    0.15 : bonus per well-timed digest (released at high attention).
    """
    w_value: float = 1.0
    w_interrupt: float = 1.0
    w_miss: float = 3.0
    w_stale: float = 0.02
    w_fatigue: float = 0.3
    w_digest: float = 0.15


@dataclass
class EnvConfig:
    """Simulator settings."""
    profile: str = "student"            # simulated user profile (user_model.py)
    tick_minutes: int = 5               # one tick = 5 minutes
    ticks_per_day: int = 288            # full 24h day
    weekday_prob: float = 5.0 / 7.0     # day type sampled per episode
    activity_stickiness: float = 0.85   # P(stay in current activity) per tick
    attention_noise: float = 0.08       # hidden attention jitter around mean
    obs_attention_noise: float = 0.15   # sensor noise on the agent's estimate
    obs_activity_flip: float = 0.15     # P(observed activity is wrong)
    daily_arrivals: float = 110.0       # expected notifications per day
    urgent_threshold: float = 0.70      # urgency >= this counts as urgent
    urgent_message_prob: float = 0.10   # P(a message is call-like urgent)
    relevance_noise: float = 0.10
    relevant_threshold: float = 0.50    # for the "relevant delivered" metric
    defer_ticks: int = 3                # deferred item is re-presented after 15 min
    max_pending: int = 20               # deferred + digest capacity
    digest_limit: int = 6               # digest auto-released at this size
    digest_max_age: int = 18            # ... or after 90 minutes
    digest_good_attention: float = 0.6  # release counts as well-timed above this
    digest_value_factor: float = 0.8    # items read in a digest get less attention
    same_tick_discount: float = 0.5     # extra alerts in an already-interrupted tick
    fatigue_budget: int = 6             # interruptions per rolling hour before fatigue
    value_halflife_ticks: float = 24.0  # non-urgent value halves every 2 hours
    rewards: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class AgentConfig:
    """DQN hyperparameters.

    Small MLP because the state is a 28-d hand-built feature vector; Huber loss
    and gradient clipping keep updates stable with the occasional large miss
    penalty; gamma=0.97 over *decisions* gives an effective horizon of ~33
    decisions, roughly 6-8 hours of simulated time.
    """
    hidden: Tuple[int, int] = (128, 128)
    lr: float = 5e-4
    gamma: float = 0.97
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup: int = 1_000
    train_every: int = 1
    target_tau: float = 0.005          # soft (Polyak) target update
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6        # fraction of training spent annealing
    double: bool = True
    dueling: bool = True


@dataclass
class TrainConfig:
    episodes: int = 400
    eval_every: int = 20               # greedy evaluation cadence (episodes)
    eval_episodes: int = 10
    frame_stack: int = 1
    seed: int = 0


# Fixed seed sets so that every policy is compared on identical days.
VALIDATION_SEEDS = tuple(range(5_000, 5_010))   # learning curves during training
HELDOUT_SEED_OFFSET = 10_000                      # final evaluation: 10000, 10001, ...


def to_dict(obj) -> dict:
    """Serialise a (nested) config dataclass to plain python types."""
    return asdict(obj)


__all__ = [
    "ACTIVITIES", "CATEGORIES", "ACTIONS", "DELIVER", "DEFER", "BATCH", "SUPPRESS",
    "ATTENTION_MEAN", "CategorySpec", "CATEGORY_SPECS", "RewardWeights",
    "EnvConfig", "AgentConfig", "TrainConfig", "to_dict", "replace",
    "VALIDATION_SEEDS", "HELDOUT_SEED_OFFSET",
]
