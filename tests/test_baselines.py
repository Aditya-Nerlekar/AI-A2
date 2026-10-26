"""Baselines must produce valid actions using only the observation."""
import numpy as np

from notifyrl.baselines import _tick_of_day, all_baselines
from notifyrl.env import OBS_INDEX, make_env


def test_baselines_return_valid_actions():
    env = make_env()
    for name, policy in all_baselines(seed=0).items():
        obs, _ = env.reset(seed=2)
        done = False
        while not done:
            a = policy(obs)
            assert a in (0, 1, 2, 3), name
            obs, _, done, _, _ = env.step(a)


def test_tick_decoding_roundtrip():
    for tick in (0, 1, 100, 143, 287):
        frac = tick / 288
        obs = np.zeros(28, dtype=np.float32)
        obs[OBS_INDEX["time_sin"]] = np.sin(2 * np.pi * frac)
        obs[OBS_INDEX["time_cos"]] = np.cos(2 * np.pi * frac)
        assert _tick_of_day(obs) == tick
