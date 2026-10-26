# NotifyRL — Intelligent Notification Scheduling with Deep Q-Learning

NotifyRL trains a reinforcement-learning agent to decide, for every incoming
smartphone notification, whether to **Deliver** it now, **Defer** it, **Batch**
it into a digest, or **Suppress** it. The agent balances the value of timely,
relevant information against the cost of interrupting a user who is asleep,
focused or in a meeting, while never missing time-critical alerts such as OTPs
or meeting reminders.

> **Honesty note.** No real user data is used. All behaviour comes from a
> synthetic simulator whose parameters are reasoned assumptions (documented in
> `notifyrl/config.py` and `notifyrl/user_model.py`). Results describe this
> simulated world, not real users.

AI Course — Assignment 2: *Real-Life Use Cases of Reinforcement Learning*.
All code (environment, DQN, replay buffer, baselines, evaluation) is original;
no RL libraries such as stable-baselines3 are used.

---

## 1. RL formulation (summary)

| Element | Definition |
|---|---|
| **Decision epoch** | Event-driven: one decision per notification needing action (new arrival or a deferred item coming back). Simulated clock: 5-minute ticks, 288 per day; ≈112 decisions/day on average. |
| **State** $s\in[-1,1]^{28}$ | sin/cos time of day (2), weekday flag (1), *observed* activity one-hot (6), *noisy* attention estimate (1), current item urgency (1), category one-hot (7), sender importance (1), item age (1), time to deadline (1), pending-queue length (1), digest length (1), oldest pending age (1), max pending urgency (1), time since last alert (1), alerts in last hour (1), fraction of day left (1). |
| **Actions** | `Discrete(4)`: 0 Deliver, 1 Defer (re-present after 15 min), 2 Batch (digest), 3 Suppress |
| **Reward** | value of delivery − interruption cost − missed-urgent penalty − staleness − fatigue + digest bonus (formula below) |
| **Episode** | One simulated day (00:00–24:00); terminates at day end; γ = 0.97 per decision (effective horizon ≈ 33 decisions ≈ 6–8 h) |
| **Algorithm** | DQN from scratch in PyTorch, with Double and Dueling variants |

Reward credited to a decision (all reward that accrues until the next decision):

$$
r = w_v\,\rho_n f_n\,(0.25+0.75a) \;-\; w_i(1-a)\,\sigma_n d \;-\; w_m\!\!\sum_{k\in\text{missed}}\!\! u_k \;-\; w_s\!\!\sum_{\text{ticks}}\sum_{k\in\text{held}}\! u_k \;-\; w_f\max(0, I_{1h}+1-B) \;+\; w_d\,\mathbb{1}[a\ge 0.6]
$$

ρ relevance, f freshness (half-life 2 h for non-urgent items), a hidden attention,
σ salience, d = 0.5 for a second alert in the same tick, u urgency, I₁ₕ alerts in
the past hour, B = 6. Weights (`config.RewardWeights`): w_v = 1, w_i = 1, w_m = 3,
w_s = 0.02, w_f = 0.3, w_d = 0.15. A digest release delivers all held items at
80 % value for a single interruption with salience 0.6. Rationale for each weight
is in the `RewardWeights` docstring.

### Simulator
* **User** (`user_model.py`): hidden activity ∈ {sleeping, focused, meeting,
  commuting, idle, on_phone} follows a time-inhomogeneous Markov chain with
  persistence 0.85 per tick and hour-of-day targets that differ for weekdays and
  weekends. Profiles: `student` (training) and `night_shift` (generalisation test).
  Hidden attention = activity mean + N(0, 0.08). The agent sees attention + N(0, 0.15)
  and an activity label that is wrong 15 % of the time → partial observability.
* **Notifications** (`notifications.py`): non-homogeneous Poisson arrivals
  (≈110/day, world-traffic curve peaking in the evening) over 7 categories with
  category-specific urgency, relevance, salience and lifetime. OTPs (10 min),
  calendar reminders (15 min) and 10 % of chat messages ("calls") are urgent and
  expire hard at their deadline (≈16 urgent items/day).
* **Queue**: deferred + digest capacity 20 (overflow drops the oldest item);
  the digest is released when the user is idle/on the phone, when it holds 6
  items, or after 90 min (not while asleep).

## 2. Installation

Python ≥ 3.9. CPU only.

```bash
git clone [GITHUB_LINK] notifyrl && cd notifyrl
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # or: pip install -e ".[test]"
# optional, smaller download: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dependencies: numpy ≥ 1.24, torch ≥ 2.1, gymnasium ≥ 0.29, matplotlib ≥ 3.7,
pandas ≥ 2.0, pytest ≥ 7.4.

## 3. Running

Reproduce everything (tests → training → evaluation → figures → LaTeX tables):

```bash
bash scripts/run_all.sh            # full pipeline
bash scripts/run_all.sh --quick    # smoke run with fewer seeds/episodes
```

Individual stages:

```bash
python -m pytest -q                                          # unit tests
python -m notifyrl.train --variant dueling_double --seeds 0 1 2 3 4 --tag dueling_double
python -m notifyrl.train --variant vanilla --seeds 0 1 2 3 4 --tag vanilla
python -m notifyrl.train --variant dueling_double --frame-stack 4 --seeds 0 1 2 --tag fs4
python -m notifyrl.train --reward w_interrupt=2.0 --seeds 0 1 --tag rw_interrupt_x2
python -m notifyrl.evaluate --episodes 100 --models "models/*.pt"   # baselines + agents
python -m notifyrl.evaluate --baselines-only                         # no torch needed
python -m notifyrl.plots                                             # figures + tables
```

`python -m notifyrl.train --help` lists every option (variant, seeds, episodes,
frame stack, user profile, reward-weight overrides, fine-tuning).

**Expected runtime.** The simulator runs ≈ 10 ms per simulated day, so cost
is dominated by DQN updates (≈ 45k gradient steps per 400-episode run). The
full pipeline is 29 training runs + 3 short fine-tunes; on a recent laptop CPU
this is estimated at roughly 25–35 minutes (`--quick`: under 10 minutes). These
are estimates — please record your measured time (printed at the end of
`run_all.sh`) here.

## 4. Reproducibility

* Training run with seed *s* uses simulated days `s·100000 + episode`.
* Learning curves are measured on fixed validation days 5000–5009.
* Final evaluation uses held-out days 10000–10099 (never seen in training).
* Each day uses independent RNG streams for the hidden user, the notification
  arrivals and the sensor noise, so **every policy faces exactly the same days**
  (paired comparison), regardless of which actions it takes.
* `torch.manual_seed`, `numpy` seeds and single-threaded torch are set in `train.py`.

## 5. Outputs

| Path | Content |
|---|---|
| `models/<tag>_s<seed>.pt` | trained agents (weights, config, metadata) |
| `results/curves/<tag>_s<seed>.csv` | per-episode training log + greedy validation return |
| `results/eval_episodes*.csv` | per-episode metrics for every policy |
| `results/eval_summary*.csv` | mean, std, 95 % CI per policy × metric (`_night` = night-shift profile) |
| `results/hourly_actions*.csv` | action counts by hour of day |
| `results/figures/fig*.pdf/png` | report figures |
| `report/generated/*.tex` | result tables auto-filled for the LaTeX report |

Experiment tags: `vanilla`, `double`, `dueling_double` (main agent), `fs4`
(4-frame observation stack), `rw_*` (reward sensitivity), `night_native`,
`finetune_night` (generalisation).

## 6. Baselines (comparison only — not the solution)

1. **Always-Deliver** — default phone behaviour.
2. **Random** — uniform over the 4 actions.
3. **Fixed schedule** — urgent now, everything else released in a slot at the top of each hour.
4. **Do-Not-Disturb** — deliver if urgent or the user looks idle/on the phone; otherwise batch (promotions suppressed).
5. **Attention + urgency threshold** — deliver if urgent or urgency + attention ≥ 1, batch the middle band, suppress low-value items.

All baselines see exactly the agent's noisy observation.

## 7. Repository layout

```
notifyrl/
  README.md  requirements.txt  pyproject.toml
  notifyrl/
    config.py         hyperparameters, reward weights, category specs, seed sets
    user_model.py     hidden activity/attention Markov model, user profiles
    notifications.py  non-homogeneous Poisson notification generator
    env.py            Gymnasium env (event-driven), metrics, frame-stack wrapper
    replay.py         ring-buffer experience replay
    dqn.py            Q-network (plain/dueling), DQN/Double DQN agent
    baselines.py      5 reference policies
    train.py          training CLI
    evaluate.py       held-out evaluation CLI
    plots.py          figures + LaTeX result tables
  tests/              pytest suite (env, baselines, DQN)
  scripts/run_all.sh  end-to-end reproduction
  results/  models/   outputs
  report/             LaTeX report (IEEEtran) + bibliography
```

## 8. Report

`report/main.tex` (IEEEtran conference, two-column) + `report/references.bib`.
It pulls figures from `results/figures/` and table rows from `report/generated/`
(both written by `python -m notifyrl.plots`), so every number in the tables comes
from your own runs; anything not yet produced shows as a red **TODO**.

* Local: `cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`
* Overleaf: upload `main.tex`, `references.bib`, the `generated/` folder, and the
  PDFs from `results/figures/` into a `figures/` folder; compile with pdfLaTeX.

## 9. Limitations

The simulator encodes our assumptions about attention, urgency and annoyance;
real users differ, and engagement is modelled as an expected value rather than
measured behaviour. Policies trained here would need offline evaluation on
logged data and careful online A/B testing before any real deployment.
