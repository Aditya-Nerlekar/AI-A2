"""Sanity checks for the simulator (no torch required)."""
import numpy as np
import pytest

from notifyrl.config import ACTIVITIES, BATCH, DEFER, DELIVER, SUPPRESS, EnvConfig
from notifyrl.env import OBS_DIM, OBS_INDEX, make_env
from notifyrl.notifications import NotificationStream
from notifyrl.user_model import PROFILES, SimulatedUser


def rollout(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    rewards, observations, done = [], [obs], False
    while not done:
        obs, r, done, trunc, info = env.step(policy(obs))
        assert not trunc
        rewards.append(r)
        observations.append(obs)
    return rewards, observations, info


def test_observation_shape_and_bounds():
    env = make_env()
    assert env.observation_space.shape == (OBS_DIM,)
    assert OBS_DIM == 28
    _, observations, _ = rollout(env, lambda o: DELIVER, seed=3)
    arr = np.stack(observations)
    assert arr.shape[1] == OBS_DIM
    assert np.all(np.isfinite(arr))
    assert arr.min() >= -1.0 and arr.max() <= 1.0


def test_one_hot_blocks_are_valid():
    env = make_env()
    obs, _ = env.reset(seed=11)
    a0 = OBS_INDEX["activity"]
    c0 = OBS_INDEX["category"]
    assert obs[a0:a0 + len(ACTIVITIES)].sum() == pytest.approx(1.0)
    assert obs[c0:c0 + 7].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("action", [DELIVER, DEFER, BATCH, SUPPRESS])
def test_episode_terminates_for_every_constant_policy(action):
    env = make_env()
    rewards, _, info = rollout(env, lambda o: action, seed=5)
    assert 20 < len(rewards) < 2_000
    assert "episode" in info
    assert np.isfinite(info["episode"]["return"])


def test_reward_bounds_per_decision():
    """A single decision can at most gain a few units and lose a bounded amount."""
    env = make_env()
    rng = np.random.default_rng(0)
    for seed in range(5):
        rewards, _, _ = rollout(env, lambda o: int(rng.integers(4)), seed=seed)
        r = np.asarray(rewards)
        assert np.all(np.isfinite(r))
        assert r.max() < 10.0
        assert r.min() > -60.0


def test_determinism_by_seed():
    env_a, env_b = make_env(), make_env()
    pol = lambda o: int(o[OBS_INDEX["urgency"]] > 0.5)  # deterministic in obs
    ra, oa, ia = rollout(env_a, pol, seed=42)
    rb, ob, ib = rollout(env_b, pol, seed=42)
    assert ra == rb
    assert all(np.array_equal(x, y) for x, y in zip(oa, ob))
    rc, _, _ = rollout(env_a, pol, seed=43)
    assert rc != ra


def test_same_seed_gives_same_world_for_any_policy():
    """Arrivals and hidden user day must not depend on the policy."""
    env = make_env()
    _, _, i1 = rollout(env, lambda o: DELIVER, seed=7)
    arrivals_1 = env.unwrapped.stream.total
    _, _, i2 = rollout(env, lambda o: SUPPRESS, seed=7)
    assert env.unwrapped.stream.total == arrivals_1
    assert i1["episode"]["arrived"] == i2["episode"]["arrived"]


def test_suppressing_everything_misses_all_urgent():
    env = make_env()
    _, _, info = rollout(env, lambda o: SUPPRESS, seed=1)
    s = env.unwrapped.stats
    assert s["urgent_arrived"] > 0
    assert info["episode"]["urgent_miss_rate"] == pytest.approx(1.0)
    assert info["episode"]["interruptions"] == 0


def test_delivering_everything_never_misses_urgent():
    env = make_env()
    _, _, info = rollout(env, lambda o: DELIVER, seed=1)
    assert info["episode"]["urgent_miss_rate"] == 0.0
    assert info["episode"]["urgent_latency_min"] == 0.0


def test_deferring_an_otp_makes_it_expire():
    """OTP deadline (10 min) is shorter than the defer delay (15 min)."""
    env = make_env()
    otp = OBS_INDEX["category"] + 5
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = DEFER if obs[otp] > 0.5 else DELIVER
        obs, _, done, _, info = env.step(action)
    stats = env.unwrapped.stats
    assert stats["dropped_expired"] > 0


def test_frame_stack_shape():
    env = make_env(frame_stack=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4 * OBS_DIM,)
    obs, *_ = env.step(DELIVER)
    assert obs.shape == (4 * OBS_DIM,)


def test_user_profiles_are_proper_distributions():
    for name in PROFILES:
        user = SimulatedUser(EnvConfig(profile=name), np.random.default_rng(0))
        for table in user._tables.values():
            assert table.shape == (24, len(ACTIVITIES))
            assert np.allclose(table.sum(axis=1), 1.0)


def test_night_shift_sleeps_in_daytime():
    cfg = EnvConfig(profile="night_shift")
    user = SimulatedUser(cfg, np.random.default_rng(0))
    user.start_day(weekday=True)
    asleep = 0
    for t in range(12 * 11, 12 * 15):      # 11:00-15:00
        user.advance(t)
        asleep += user.activity_name == "sleeping"
    assert asleep > 0.6 * 48


def test_arrival_rate_matches_daily_budget():
    cfg = EnvConfig()
    stream = NotificationStream(cfg, np.random.default_rng(0))
    total = sum(stream.rate(t) for t in range(cfg.ticks_per_day))
    assert total == pytest.approx(cfg.daily_arrivals, rel=1e-6)
    counts = []
    for _ in range(30):
        stream.generate_day()
        counts.append(stream.total)
    assert abs(np.mean(counts) - cfg.daily_arrivals) < 10
