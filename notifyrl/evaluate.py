"""Held-out evaluation of baselines and trained agents.

Every policy is run on the *same* held-out seeds (default 10000..10099), so
all policies face identical days: same hidden user trajectory and same
notification arrivals.  Results are written to ``results/``:

    eval_episodes.csv   one row per (policy, episode) with all metrics
    eval_summary.csv    mean, std and 95% CI per policy and metric
    hourly_actions.csv  action counts by hour of day per policy

Usage
    python -m notifyrl.evaluate --models models/dqn_dueling_double_s0.pt ...
    python -m notifyrl.evaluate --baselines-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .baselines import all_baselines
from .config import HELDOUT_SEED_OFFSET, EnvConfig
from .env import make_env

METRICS = [
    "return", "urgent_miss_rate", "urgent_latency_min", "interruptions",
    "interruptions_per_awake_hour", "interruptions_focus_meeting_sleep",
    "interruptions_sleep", "relevant_delivered_rate", "nonurgent_delay_min",
    "digests", "frac_deliver", "frac_defer", "frac_batch", "frac_suppress", "decisions",
]


def run_policy(policy: Callable[[np.ndarray], int], episodes: int = 100, seed_offset: int = 10_000,
               profile: str = "student", frame_stack: int = 1,
               env_cfg: Optional[EnvConfig] = None) -> (pd.DataFrame, np.ndarray):
    """Roll out ``policy`` greedily; return per-episode metrics and an
    (24 x 4) hour-by-action count matrix."""
    cfg = env_cfg if env_cfg is not None else EnvConfig(profile=profile)
    env = make_env(cfg, frame_stack=frame_stack)
    rows: List[Dict] = []
    hourly = np.zeros((24, 4), dtype=np.int64)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs))
        summary = dict(info["episode"])
        summary["seed"] = seed_offset + ep
        rows.append(summary)
        for hour, action, _ in env.unwrapped.stats["action_log"]:
            hourly[hour, action] += 1
    return pd.DataFrame(rows), hourly


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / 95% normal-approximation CI per policy and metric."""
    out = []
    for policy, grp in df.groupby("policy", sort=False):
        for m in METRICS:
            vals = grp[m].dropna().to_numpy(dtype=float)
            n = len(vals)
            mean = float(vals.mean()) if n else np.nan
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            half = 1.96 * std / np.sqrt(n) if n > 1 else np.nan
            out.append({"policy": policy, "metric": m, "mean": mean, "std": std,
                        "ci95_low": mean - half, "ci95_high": mean + half, "n": n})
    return pd.DataFrame(out)


def load_agent_policy(path: str):
    """Load a checkpoint lazily (torch is only needed when models are given)."""
    from .dqn import DQNAgent
    agent = DQNAgent.load(path)
    return agent, agent.meta.get("frame_stack", 1)


def evaluate_all(model_paths: List[str], episodes: int, seed_offset: int, profile: str,
                 include_baselines: bool = True, label_by: str = "tag") -> Dict[str, pd.DataFrame]:
    frames, hourly_rows = [], []
    policies = []
    if include_baselines:
        for name, pol in all_baselines(seed=seed_offset).items():
            policies.append((name, pol, 1))
    for path in model_paths:
        agent, k = load_agent_policy(path)
        label = agent.meta.get(label_by) or os.path.splitext(os.path.basename(path))[0]
        policies.append((label, agent.greedy, k))
    for label, pol, k in policies:
        df, hourly = run_policy(pol, episodes, seed_offset, profile, frame_stack=k)
        df.insert(0, "policy", label)
        frames.append(df)
        for h in range(24):
            hourly_rows.append({"policy": label, "hour": h, "deliver": hourly[h, 0], "defer": hourly[h, 1],
                                "batch": hourly[h, 2], "suppress": hourly[h, 3]})
        print(f"  {label:<28s} return {df['return'].mean():8.2f} +/- {df['return'].std():.2f}   "
              f"urgent miss {df['urgent_miss_rate'].mean():.3f}   interrupts/awake-h "
              f"{df['interruptions_per_awake_hour'].mean():.2f}")
    episodes_df = pd.concat(frames, ignore_index=True)
    # multiple seeds of the same variant are pooled under one label
    return {"episodes": episodes_df, "summary": summarise(episodes_df),
            "hourly": pd.DataFrame(hourly_rows).groupby(["policy", "hour"], sort=False).sum().reset_index()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=[], help="checkpoint paths or globs")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-offset", type=int, default=HELDOUT_SEED_OFFSET)
    p.add_argument("--profile", default="student")
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="", help="suffix for output file names")
    args = p.parse_args(argv)

    paths: List[str] = []
    if not args.baselines_only:
        for pattern in args.models:
            paths.extend(sorted(glob.glob(pattern)) or [pattern])
    os.makedirs(args.out, exist_ok=True)
    print(f"Evaluating on profile={args.profile}, {args.episodes} held-out episodes "
          f"(seeds {args.seed_offset}..{args.seed_offset + args.episodes - 1})")
    res = evaluate_all(paths, args.episodes, args.seed_offset, args.profile,
                       include_baselines=not args.no_baselines)
    tag = f"_{args.tag}" if args.tag else ""
    res["episodes"].to_csv(os.path.join(args.out, f"eval_episodes{tag}.csv"), index=False)
    res["summary"].to_csv(os.path.join(args.out, f"eval_summary{tag}.csv"), index=False)
    res["hourly"].to_csv(os.path.join(args.out, f"hourly_actions{tag}.csv"), index=False)
    with open(os.path.join(args.out, f"eval_meta{tag}.json"), "w") as fh:
        json.dump({"profile": args.profile, "episodes": args.episodes,
                   "seed_offset": args.seed_offset, "models": paths}, fh, indent=2)
    print(f"wrote results to {args.out}/ (tag='{args.tag}')")


if __name__ == "__main__":
    main()
