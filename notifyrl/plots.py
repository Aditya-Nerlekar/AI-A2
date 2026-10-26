"""Produce every report figure and LaTeX table from saved results.

Reads   results/curves/*.csv, results/eval_summary*.csv, results/hourly_actions*.csv,
        models/*.pt (policy heatmap only)
Writes  results/figures/*.pdf|png and report/generated/*.tex

Anything that has not been produced yet is skipped (figures) or rendered as
"TODO" (tables), so the script is safe to run at any stage of the pipeline.

Usage:  python -m notifyrl.plots
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baselines import all_baselines  # noqa: E402
from .config import ACTIONS, ACTIVITIES, CATEGORIES, VALIDATION_SEEDS  # noqa: E402
from .env import OBS_DIM, OBS_INDEX  # noqa: E402
from .evaluate import run_policy  # noqa: E402

BASELINES = ["always_deliver", "random", "fixed_schedule", "do_not_disturb", "threshold"]
VARIANT_TAGS = ["vanilla", "double", "dueling_double"]
MAIN_TAG = "dueling_double"
REWARD_TAGS = ["rw_interrupt_x0.5", "rw_interrupt_x2", "rw_miss_x0.5", "rw_miss_x2"]
NICE = {
    "always_deliver": "Always-Deliver", "random": "Random", "fixed_schedule": "Fixed schedule",
    "do_not_disturb": "Do-Not-Disturb", "threshold": "Attn+Urg threshold",
    "vanilla": "DQN", "double": "Double DQN", "dueling_double": "Dueling+Double DQN",
    "fs4": "D+D DQN, 4-frame stack",
    "rw_interrupt_x0.5": r"$w_i\times0.5$", "rw_interrupt_x2": r"$w_i\times2$",
    "rw_miss_x0.5": r"$w_m\times0.5$", "rw_miss_x2": r"$w_m\times2$",
    "finetune_night": "Student DQN + fine-tune", "night_native": "DQN trained on night-shift",
}
SHORT = {"always_deliver": "Always", "random": "Random", "fixed_schedule": "Schedule",
         "do_not_disturb": "DND", "threshold": "Threshold", "vanilla": "DQN", "double": "Double",
         "dueling_double": "Duel+Dbl", "fs4": "+4 frames"}
TINY = {"always_deliver": "AD", "random": "Rnd", "fixed_schedule": "Sch", "do_not_disturb": "DND",
        "threshold": "Thr", "dueling_double": "DQN"}
ACTION_COLORS = ["#2a6fdb", "#f0a202", "#3aa76d", "#c4423c"]
BASE_COLOR, DQN_COLOR = "#9aa5b1", "#2a6fdb"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def _summary(results: str, tag: str = "") -> Optional[pd.DataFrame]:
    path = os.path.join(results, f"eval_summary{'_' + tag if tag else ''}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def _get(summary: Optional[pd.DataFrame], policy: str, metric: str, col: str = "mean") -> float:
    if summary is None:
        return np.nan
    row = summary[(summary.policy == policy) & (summary.metric == metric)]
    return float(row[col].iloc[0]) if len(row) else np.nan


# --------------------------------------------------------------------- figures
def fig_training_curves(results: str, out: str) -> None:
    curves = {}
    for tag in VARIANT_TAGS + ["fs4"]:
        files = sorted(glob.glob(os.path.join(results, "curves", f"{tag}_s*.csv")))
        if files:
            curves[tag] = [pd.read_csv(f).dropna(subset=["eval_return"]) for f in files]
    if not curves:
        print("  [skip] training curves: no results/curves/*.csv yet")
        return
    fig, ax = plt.subplots(figsize=(2.3, 1.3))
    for tag, dfs in curves.items():
        ep = dfs[0]["episode"].to_numpy()
        ys = np.stack([d["eval_return"].to_numpy()[:len(ep)] for d in dfs])
        m, s = ys.mean(0), ys.std(0)
        line, = ax.plot(ep, m, lw=1.1, label=SHORT[tag])
        ax.fill_between(ep, m - s, m + s, alpha=0.2, color=line.get_color())
    pols = all_baselines(seed=0)
    for name, ls in [("threshold", "--"), ("do_not_disturb", ":"), ("always_deliver", "-.")]:
        df, _ = run_policy(pols[name], episodes=len(VALIDATION_SEEDS), seed_offset=VALIDATION_SEEDS[0])
        ax.axhline(df["return"].mean(), color="k", ls=ls, lw=0.8, label=SHORT[name])
    ax.set_xlabel("training episode (day)")
    ax.set_ylabel("val. return / day")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.42), fontsize=5.5,
              columnspacing=0.6, handlelength=1.4, handletextpad=0.3)
    _save(fig, out, "fig1_training_curve")


def fig_metric_bars(results: str, out: str) -> None:
    summ = _summary(results)
    if summ is None:
        print("  [skip] metric bars: run evaluate first")
        return
    policies = [p for p in BASELINES + [MAIN_TAG] if p in set(summ.policy)]
    panels = [("return", "return / day"), ("urgent_miss_rate", "urgent miss rate"),
              ("interruptions_focus_meeting_sleep", "alerts f/m/s / day"),
              ("relevant_delivered_rate", "relevant delivered")]
    fig, axes = plt.subplots(1, 4, figsize=(4.8, 1.25))
    fig.subplots_adjust(wspace=0.5)
    for ax, (metric, title) in zip(axes, panels):
        means = [_get(summ, p, metric) for p in policies]
        err = [_get(summ, p, metric, "ci95_high") - _get(summ, p, metric) for p in policies]
        colors = [DQN_COLOR if p == MAIN_TAG else BASE_COLOR for p in policies]
        ax.bar(range(len(policies)), means, yerr=err, color=colors, capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([TINY[p] for p in policies], fontsize=5.5, rotation=60)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
    _save(fig, out, "fig2_metrics_vs_baselines")


def fig_hourly_actions(results: str, out: str) -> None:
    path = os.path.join(results, "hourly_actions.csv")
    if not os.path.exists(path):
        print("  [skip] hourly actions: run evaluate first")
        return
    df = pd.read_csv(path)
    if MAIN_TAG not in set(df.policy):
        print(f"  [skip] hourly actions: no '{MAIN_TAG}' agent evaluated yet")
        return
    shown = [MAIN_TAG]
    fig, axes = plt.subplots(1, 1, figsize=(2.35, 1.35), squeeze=False)
    for ax, pol in zip(axes[0], shown):
        d = df[df.policy == pol].sort_values("hour")
        counts = d[list(ACTIONS)].to_numpy(dtype=float)
        frac = counts / np.maximum(counts.sum(1, keepdims=True), 1)
        bottom = np.zeros(24)
        for k, act in enumerate(ACTIONS):
            ax.bar(d.hour, frac[:, k], bottom=bottom, color=ACTION_COLORS[k], width=0.9, label=act)
            bottom += frac[:, k]
        ax.set_xlim(-0.6, 23.6)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("hour of day")
    axes[0][0].set_ylabel("share of decisions")
    axes[0][-1].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.33),
                       columnspacing=0.6, handlelength=1.0, handletextpad=0.3)
    _save(fig, out, "fig3_actions_by_hour")


def _canonical_obs(attention: float, urgency: float, activity: str, category: str = "messaging",
                   hour: float = 14.0) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    frac = hour / 24.0
    obs[OBS_INDEX["time_sin"]], obs[OBS_INDEX["time_cos"]] = np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac)
    obs[OBS_INDEX["weekday"]] = 1.0
    obs[OBS_INDEX["activity"] + ACTIVITIES.index(activity)] = 1.0
    obs[OBS_INDEX["attention_est"]] = attention
    obs[OBS_INDEX["urgency"]] = urgency
    obs[OBS_INDEX["category"] + CATEGORIES.index(category)] = 1.0
    obs[OBS_INDEX["importance"]] = 0.5
    obs[OBS_INDEX["time_to_deadline"]] = 1.0
    obs[OBS_INDEX["since_interrupt"]] = 1.0
    obs[OBS_INDEX["day_remaining"]] = 1.0 - frac
    return obs


def fig_policy_heatmap(models: str, out: str) -> None:
    path = os.path.join(models, f"{MAIN_TAG}_s0.pt")
    if not os.path.exists(path):
        print(f"  [skip] policy heatmap: {path} not found")
        return
    from matplotlib.colors import ListedColormap
    from .dqn import DQNAgent
    agent = DQNAgent.load(path)
    grid = np.linspace(0, 1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(2.4, 1.35), sharey=True)
    for ax, act in zip(axes, ("focused", "idle")):
        obs = np.stack([_canonical_obs(a, u, act) for u in grid for a in grid])
        choice = agent.q_values(obs).argmax(1).reshape(len(grid), len(grid))
        ax.imshow(choice, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
                  cmap=ListedColormap(ACTION_COLORS), vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set_title(f"observed: {act}", fontsize=6.5)
        ax.set_xlabel("attention est.")
    axes[0].set_ylabel("urgency")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ACTION_COLORS]
    fig.legend(handles, ACTIONS, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.97),
               columnspacing=0.6, handlelength=1.0, handletextpad=0.3)
    _save(fig, out, "fig4_policy_heatmap")


def fig_reward_sensitivity(results: str, out: str) -> None:
    """Behaviour of agents retrained with halved/doubled reward weights.

    Bars: alerts per day while focused/in a meeting/asleep; labels on top:
    urgent-miss rate.  (DQN variant ablation is reported in the main table.)
    """
    summ = _summary(results)
    tags = [t for t in [MAIN_TAG] + REWARD_TAGS if summ is not None and t in set(summ.policy)]
    if len(tags) < 2:
        print("  [skip] reward sensitivity: rw_* agents not evaluated yet")
        return
    fig, ax = plt.subplots(figsize=(2.3, 1.35))
    x = np.arange(len(tags))
    alerts = [_get(summ, t, "interruptions_focus_meeting_sleep") for t in tags]
    err = [_get(summ, t, "interruptions_focus_meeting_sleep", "ci95_high") - a for t, a in zip(tags, alerts)]
    ax.bar(x, alerts, yerr=err, color=[DQN_COLOR] + ["#7d8b99"] * (len(tags) - 1), capsize=2,
           error_kw={"lw": 0.8})
    for xi, t, a in zip(x, tags, alerts):
        miss = 100 * _get(summ, t, "urgent_miss_rate")
        ax.text(xi, a * 1.04 + 0.3, f"{miss:.1f}%", ha="center", va="bottom", fontsize=5.5, color="#c4423c")
    ax.set_xticks(x)
    ax.set_xticklabels(["default" if t == MAIN_TAG else NICE[t] for t in tags], fontsize=6, rotation=35, ha="right")
    ax.set_ylabel("alerts f/m/s / day")
    ax.set_ylim(top=max(alerts) * 1.3 + 1)
    ax.text(0.99, 0.98, "red: urgent miss rate", transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="#c4423c")
    _save(fig, out, "fig5_reward_sensitivity")


# ---------------------------------------------------------------------- tables
def _fmt(summ, pol, metric, scale=1.0, digits=1, ci=False) -> str:
    v = _get(summ, pol, metric)
    if not np.isfinite(v):
        return r"\todo{}"
    txt = f"{v * scale:.{digits}f}".replace("-", "$-$")
    if ci:
        half = _get(summ, pol, metric, "ci95_high") - v
        txt += f"$\\pm${half * scale:.{digits}f}"
    return txt


def write_latex_tables(results: str, report_dir: str) -> None:
    gen = os.path.join(report_dir, "generated")
    os.makedirs(gen, exist_ok=True)
    summ = _summary(results)
    rows = BASELINES + VARIANT_TAGS + ["fs4"]
    lines = []
    for p in rows:
        if p == "vanilla":
            lines.append(r"\midrule")
        lines.append(" & ".join([
            NICE[p].replace("+", "{+}"), _fmt(summ, p, "return", ci=True),
            _fmt(summ, p, "urgent_miss_rate", 100, 1), _fmt(summ, p, "urgent_latency_min", 1, 1),
            _fmt(summ, p, "interruptions_per_awake_hour", 1, 2),
            _fmt(summ, p, "interruptions_focus_meeting_sleep", 1, 1),
            _fmt(summ, p, "relevant_delivered_rate", 100, 1), _fmt(summ, p, "nonurgent_delay_min", 1, 0),
        ]) + r" \\")
    with open(os.path.join(gen, "main_table_rows.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by notifyrl.plots -- do not edit by hand\n" + "\n".join(lines) + "\n")

    night = _summary(results, "night")
    lines = []
    for p, label in [("do_not_disturb", "Do-Not-Disturb"), ("threshold", "Attn+Urg threshold"),
                     (MAIN_TAG, "Student DQN, zero-shot"), ("finetune_night", "Student DQN + 100-ep fine-tune"),
                     ("night_native", "DQN trained on night-shift")]:
        lines.append(" & ".join([label, _fmt(night, p, "return", ci=True), _fmt(night, p, "urgent_miss_rate", 100, 1),
                                 _fmt(night, p, "interruptions_focus_meeting_sleep", 1, 1)]) + r" \\")
    with open(os.path.join(gen, "generalisation_rows.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by notifyrl.plots -- do not edit by hand\n" + "\n".join(lines) + "\n")
    print(f"  wrote {gen}/main_table_rows.tex and generalisation_rows.tex")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results")
    p.add_argument("--models", default="models")
    p.add_argument("--report", default="report")
    args = p.parse_args(argv)
    out = os.path.join(args.results, "figures")
    print("Generating figures ...")
    fig_training_curves(args.results, out)
    fig_metric_bars(args.results, out)
    fig_hourly_actions(args.results, out)
    fig_policy_heatmap(args.models, out)
    fig_reward_sensitivity(args.results, out)
    write_latex_tables(args.results, args.report)


if __name__ == "__main__":
    main()
