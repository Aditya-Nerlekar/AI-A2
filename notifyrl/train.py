"""Train DQN agents on the NotifyRL simulator.

Examples
    # main agent (Dueling + Double DQN), 5 seeds
    python -m notifyrl.train --variant dueling_double --seeds 0 1 2 3 4 --tag dueling_double

    # partial-observability experiment: stack the last 4 observations
    python -m notifyrl.train --variant dueling_double --frame-stack 4 --seeds 0 1 2 --tag fs4

    # reward sensitivity: double the interruption weight
    python -m notifyrl.train --reward w_interrupt=2.0 --seeds 0 1 --tag rw_interrupt_x2

    # fine-tune a student agent on the night-shift profile
    python -m notifyrl.train --profile night_shift --init-from models/dueling_double_s0.pt \
        --episodes 100 --eps-start 0.3 --seeds 0 --tag finetune_night

Outputs: models/<tag>_s<seed>.pt and results/curves/<tag>_s<seed>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .config import AgentConfig, EnvConfig, RewardWeights, TrainConfig, VALIDATION_SEEDS
from .dqn import DQNAgent
from .env import make_env

VARIANTS: Dict[str, Dict[str, bool]] = {
    "vanilla": {"double": False, "dueling": False},
    "double": {"double": True, "dueling": False},
    "dueling_double": {"double": True, "dueling": True},
}
EST_DECISIONS_PER_EPISODE = 120                 # for the epsilon schedule


def greedy_return(agent: DQNAgent, env_cfg: EnvConfig, frame_stack: int, seeds: List[int]) -> float:
    env = make_env(env_cfg, frame_stack=frame_stack)
    total = 0.0
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            obs, r, done, _, _ = env.step(agent.greedy(obs))
            total += r
    return total / len(seeds)


def train_one(seed: int, variant: str, env_cfg: EnvConfig, train_cfg: TrainConfig,
              agent_cfg: AgentConfig, tag: str, models_dir: str, curves_dir: str,
              init_from: str = "", verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(env_cfg, frame_stack=train_cfg.frame_stack)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, env.action_space.n, agent_cfg, seed=seed,
                     total_steps=train_cfg.episodes * EST_DECISIONS_PER_EPISODE)
    if init_from:
        src = DQNAgent.load(init_from)
        if src.obs_dim != obs_dim:
            raise ValueError("checkpoint observation size does not match this environment")
        agent.online.load_state_dict(src.online.state_dict())
        agent.target.load_state_dict(src.online.state_dict())

    log = []
    t0 = time.time()
    for ep in range(train_cfg.episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        done, ep_ret, losses = False, 0.0, []
        while not done:
            action = agent.act(obs, explore=True)
            nxt, reward, done, _, info = env.step(action)
            loss = agent.observe(obs, action, reward, nxt, done)
            if loss is not None:
                losses.append(loss)
            obs, ep_ret = nxt, ep_ret + reward
        row = {"episode": ep + 1, "train_return": ep_ret, "epsilon": agent.epsilon(),
               "loss": float(np.mean(losses)) if losses else np.nan,
               "urgent_miss_rate": info["episode"]["urgent_miss_rate"], "eval_return": np.nan}
        if (ep + 1) % train_cfg.eval_every == 0 or ep == 0:
            row["eval_return"] = greedy_return(agent, env_cfg, train_cfg.frame_stack,
                                               list(VALIDATION_SEEDS[:train_cfg.eval_episodes]))
            if verbose:
                print(f"  [{tag} s{seed}] ep {ep + 1:4d}  train {ep_ret:7.2f}  eval {row['eval_return']:7.2f}"
                      f"  eps {agent.epsilon():.3f}  loss {row['loss']:.4f}  ({time.time() - t0:.0f}s)")
        log.append(row)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    meta = {"tag": tag, "variant": variant, "seed": seed, "profile": env_cfg.profile,
            "frame_stack": train_cfg.frame_stack, "episodes": train_cfg.episodes,
            "reward_weights": asdict(env_cfg.rewards), "init_from": init_from,
            "train_seconds": time.time() - t0}
    agent.save(os.path.join(models_dir, f"{tag}_s{seed}.pt"), extra=meta)
    pd.DataFrame(log).to_csv(os.path.join(curves_dir, f"{tag}_s{seed}.csv"), index=False)
    return meta


def parse_reward_overrides(pairs: List[str]) -> Dict[str, float]:
    out = {}
    valid = set(asdict(RewardWeights()))
    for item in pairs:
        key, _, val = item.partition("=")
        if key not in valid:
            raise SystemExit(f"unknown reward weight {key!r}; valid: {sorted(valid)}")
        out[key] = float(val)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=list(VARIANTS), default="dueling_double")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--profile", default="student")
    p.add_argument("--reward", nargs="*", default=[], metavar="NAME=VALUE",
                   help="override reward weights, e.g. w_interrupt=2.0")
    p.add_argument("--init-from", default="", help="checkpoint to fine-tune from")
    p.add_argument("--eps-start", type=float, default=None)
    p.add_argument("--tag", default=None, help="experiment name (default: variant)")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--curves-dir", default=os.path.join("results", "curves"))
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    rewards = replace(RewardWeights(), **parse_reward_overrides(args.reward))
    env_cfg = EnvConfig(profile=args.profile, rewards=rewards)
    agent_cfg = replace(AgentConfig(), **VARIANTS[args.variant])
    if args.eps_start is not None:
        agent_cfg = replace(agent_cfg, eps_start=args.eps_start)
    train_cfg = TrainConfig(episodes=args.episodes, frame_stack=args.frame_stack)
    tag = args.tag or args.variant

    print(f"Training {tag}: variant={args.variant} profile={args.profile} frame_stack={args.frame_stack} "
          f"episodes={args.episodes} seeds={args.seeds} rewards={asdict(rewards)}")
    metas = [train_one(s, args.variant, env_cfg, train_cfg, agent_cfg, tag, args.models_dir,
                       args.curves_dir, init_from=args.init_from.replace("{seed}", str(s)))
             for s in args.seeds]
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"train_meta_{tag}.json"), "w") as fh:
        json.dump(metas, fh, indent=2)
    print(f"done: {tag} ({sum(m['train_seconds'] for m in metas):.0f}s total)")


if __name__ == "__main__":
    main()
