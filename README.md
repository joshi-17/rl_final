# rl_final — Smart Energy Management RL: MLOps Edition

Q-Learning agent that learns to manage battery-powered device power levels
adaptively based on task demand. Includes full MLOps support: YAML-driven
experiments, versioned policies, experiment tracking, and a real-world
monitoring plan.

---

## File Structure

```
rl_final/
├── README.md                    ← This file
├── environment.py               ← Smart Energy RL environment
├── agent.py                     ← Tabular Q-Learning agent
├── train.py                     ← Config-driven training + MLOps tracking
├── utils.py                     ← Shared helpers (moving average, logging)
├── baseline.py                  ← Fixed medium-power baseline agent
├── compare.py                   ← Head-to-head evaluation + plots
├── configs/
│   ├── qlearning_v1.yaml        ← Experiment exp-qlearning-1 (baseline)
│   └── qlearning_v2.yaml        ← Experiment exp-qlearning-2 (tuned)
├── results.csv                  ← Experiment tracking (auto-generated)
├── logs/
│   ├── run_1.json               ← Per-episode log for run_1
│   └── run_<timestamp>.json     ← Auto-generated per run
├── plots/
│   ├── reward_vs_episodes.png
│   └── battery_vs_time.png
├── policy_v1.pkl                ← Saved policy from exp-qlearning-1
└── policy_v2.pkl                ← Saved policy from exp-qlearning-2
```

---

## Versioning

Each experiment is a separate Git commit tagged with its experiment name.

| Tag | Config | Description |
|-----|--------|-------------|
| `exp-qlearning-1` | `configs/qlearning_v1.yaml` | Baseline: 500 eps, lr=0.1, γ=0.95, decay=0.995 |
| `exp-qlearning-2` | `configs/qlearning_v2.yaml` | Tuned: 1000 eps, lr=0.15, γ=0.99, decay=0.998 |

To check out a specific experiment:
```bash
git checkout exp-qlearning-1   # restore exact state of v1 run
git checkout exp-qlearning-2   # restore exact state of v2 run
git checkout main              # return to latest
```

---

## Experiment Tracking

Every training run appends one row to **results.csv** and writes a full
per-episode log to **logs/<run_id>.json**.

### results.csv columns

| Column | Description |
|--------|-------------|
| `run_id` | Unique identifier for this run (e.g. `run_1`, `run_1718480463`) |
| `episodes` | Total training episodes |
| `avg_reward` | Mean total reward over the last 100 episodes |
| `avg_battery_remaining` | Mean battery % remaining at episode end (last 100 eps) |
| `avg_drain_per_step` | Mean battery % drained per timestep — energy efficiency metric |
| `avg_task_score` | Mean task completion score per timestep — task success rate |
| `epsilon` | Initial exploration rate used in this run |
| `learning_rate` | Alpha (α) used in this run |

### logs/<run_id>.json structure

```json
{
  "run_id": "run_1",
  "experiment_tags": ["exp-qlearning-1", "exp-qlearning-2"],
  "n_episodes": 500,
  "episode_rewards":         [12.5, 14.2, ...],
  "episode_battery":         [0.0, 2.1, ...],
  "episode_drain_per_step":  [5.0, 4.8, ...],
  "episode_task_scores":     [7.2, 8.1, ...]
}
```

`avg_drain_per_step` is the energy-domain equivalent of "average waiting time"
in traffic systems — it measures how efficiently the agent uses battery per
decision step. Lower drain with equal or higher task score = better policy.

---

## Reproducibility

All random seeds are fixed in the YAML config and passed to both the
environment and NumPy. Given the same config, every run produces the
same Q-table, rewards, and saved policy.

### Reproduce exp-qlearning-1 (baseline)

```bash
git clone <repo-url>
cd rl_final
python train.py --config configs/qlearning_v1.yaml --run_id run_1
```

Expected output (last 100-episode window):
- avg reward ≈ 155
- avg battery remaining ≈ 0 %
- avg drain per step ≈ 5.0 %
- Policy saved to: `policy_v1.pkl`

### Reproduce exp-qlearning-2 (tuned)

```bash
python train.py --config configs/qlearning_v2.yaml --run_id run_2
```

Expected output (last 100-episode window):
- avg reward ≈ 160+ (more training + better exploration)
- avg drain per step ≈ lower than v1 (slower decay lets agent learn better conservation)
- Policy saved to: `policy_v2.pkl`

### Reproduce comparison plots

```bash
python compare.py --load_policy policy_v1.pkl --episodes 200
```

Anyone can clone the repo and run the same experiment — results are
deterministic because all seeds are pinned in the YAML configs.

---

## How to Run

### Step 1 — Train the RL agent (v1)
```bash
python train.py --config configs/qlearning_v1.yaml --run_id run_1
```

### Step 2 — Train the tuned agent (v2)
```bash
python train.py --config configs/qlearning_v2.yaml --run_id run_2
```

### Step 3 — Compare RL vs baseline with plots
```bash
python compare.py --load_policy policy_v1.pkl --episodes 200
```

### Step 4 — Run baseline only
```bash
python baseline.py --episodes 200
```

---

## Comparison Table

| Metric | Baseline | RL Policy (v1) | RL Policy (v2) |
|--------|----------|----------------|----------------|
| Strategy | Always medium power | Q-learning adaptive | Q-learning tuned |
| Avg reward | ~130 | ~155 | ~160+ |
| Avg battery remaining | ~0 % | ~0 % | ~0 % |
| Avg drain per step | ~5 % (fixed) | adaptive | lower (better conserved) |
| Adapts to battery state | No | Yes | Yes |
| Adapts to task demand | No | Yes | Yes |

---

## Monitoring Plan

*Design only — describes what would be monitored if this policy were deployed
on a real battery-powered device or IoT edge node.*

If deployed in a real-world smart energy management system, we would monitor
the following signals continuously:

**Energy efficiency metrics:**
We would track `avg_drain_per_step` in real time — the average battery
percentage consumed per decision cycle. A sudden increase signals the policy
is over-powering tasks unnecessarily, wasting energy. A sustained decrease
combined with rising task-failure rates would indicate under-powering.

**Task completion quality:**
`avg_task_score` per hour would be monitored against a minimum SLA threshold.
If the score drops below the baseline's average (≈7.5 per step), it indicates
the policy is making poor power-level choices for the incoming task mix.

**Battery depletion events:**
We would count how many episodes (device cycles) end with battery at 0 %
before the scheduled task queue is complete — equivalent to a "device timeout."
A rising depletion rate indicates the policy is not conserving charge for
high-demand bursts.

**Distribution shift detection:**
Real-world task demand distributions differ from the uniform distribution
used in training. We would compare the live task-demand histogram against the
training distribution weekly. If the Kullback-Leibler divergence exceeds a
threshold (e.g., KL > 0.2), an automated re-training pipeline would be
triggered with the new task distribution.

**Policy drift alert:**
If any of the above metrics degrade by more than 15 % relative to the
30-day rolling baseline, an alert would be fired to trigger a policy review
or rollback to the last known-good `policy_v1.pkl` checkpoint.

---

## Analysis

### When RL Performs Better
- **Mixed workloads** — adapts power level to demand instead of always
  using medium power. On low-demand tasks the RL agent saves battery by
  choosing low power; on high-demand tasks it uses high power when battery
  permits.
- **Low-battery situations** — the RL agent conserves charge and avoids
  draining the last percentage points on low-value tasks.

### Failure Cases
- **Workload spikes** — a sustained burst of high-demand tasks can deplete
  battery faster than the policy anticipates.
- **Rare states** — (low battery, high demand) may be under-explored if
  training episodes are short.
- **Distribution shift** — if real-world task probabilities differ from
  training, the policy degrades without online re-training.

---

## SDG Alignment

### SDG 7 — Affordable and Clean Energy
The RL agent reduces unnecessary energy consumption by matching power output
to actual task demand, extending battery life and reducing electricity footprint.

### SDG 11 — Sustainable Cities and Communities
Smart power management at the device level reduces the resource footprint of
IoT and edge computing deployments, enabling denser and more sustainable
smart-city infrastructure.
