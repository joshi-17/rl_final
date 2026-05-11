# Smart Energy Management via Reinforcement Learning
## Final Evaluation Report

**Course:** Responsible AI & MLOps — Applied AI Track  
**Date:** 16 May 2026  
**Algorithm:** Tabular Q-Learning  
**Domain:** IoT / Edge device adaptive power management  
**SDGs:** SDG 7 (Affordable and Clean Energy), SDG 11 (Sustainable Cities)

---

## 1. Problem Statement

Battery-powered IoT and edge computing devices must make repeated decisions about
how much power to allocate to incoming tasks. A naive device always runs at a fixed
medium power level, wasting energy on easy tasks and risking poor performance on
high-demand workloads. This project trains a Q-learning agent that **adaptively
selects power level** (low / medium / high) based on current battery state and
incoming task demand, maximising cumulative reward while extending battery life.

This is directly analogous to a traffic signal problem: just as fixed-timer signals
waste green time when roads are empty, fixed-power devices waste energy when tasks
are light. Reinforcement learning enables both systems to adapt to real conditions.

---

## 2. Simulator Description

The `EnergyEnvironment` simulates a battery-powered device receiving tasks of varying
computational demand.

| Component | Details |
|-----------|---------|
| **State** | (battery_level, task_demand), each discretised to {low, medium, high} → 9 states |
| **Actions** | 0 = low_power (2% drain), 1 = medium_power (5% drain), 2 = high_power (10% drain) |
| **Reward** | task_completion_score − 0.5 × drain (matching power to demand = best score) |
| **Terminal** | Battery reaches 0% |
| **Task demand** | Sampled uniformly at random each step |

Task completion scoring penalises mismatches — using low power on a high-demand
task scores only 1 point; a perfect match scores 10.

---

## 3. RL Algorithm — Tabular Q-Learning

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

**Key design choices:**

- **Tabular Q-table:** 3 battery levels × 3 task demands × 3 actions = 27 Q-values.
  Small enough to train from scratch in seconds, fully interpretable.
- **Epsilon-greedy exploration:** Starts at ε=1.0 (pure random), decays to ε=0.01,
  then switches to pure exploitation for evaluation.
- **Two experiments:** v1 (500 eps, lr=0.1, γ=0.95) vs v2 (1000 eps, lr=0.15, γ=0.99)

**Experiment configs (YAML-driven MLOps):**

| Param | exp-qlearning-1 (v1) | exp-qlearning-2 (v2) |
|-------|----------------------|----------------------|
| Episodes | 500 | 1000 |
| Learning rate (α) | 0.10 | 0.15 |
| Discount (γ) | 0.95 | 0.99 |
| ε decay | 0.995 | 0.998 |

---

## 4. MLOps Infrastructure

| Component | Implementation |
|-----------|----------------|
| Config management | YAML files in `configs/` |
| Experiment tracking | `results.csv` (appended per run) + `logs/<run_id>.json` |
| Policy versioning | `policy_v1.pkl`, `policy_v2.pkl` (pickle) |
| Git tagging | `exp-qlearning-1`, `exp-qlearning-2` for reproducibility |
| Reproducibility | Pinned RNG seeds in YAML; same config → same Q-table |
| Deployment | Docker + docker-compose (see Section 7) |

---

## 5. Comparison Table: Fixed-Power Baseline vs RL Policy

The **fixed-power baseline** always chooses medium power (action=1) regardless of
battery state or task demand. The **RL policy** (exp-qlearning-1, policy_v1.pkl)
was evaluated over 200 episodes with seed=99 after training for 500 episodes.

| Metric | Fixed-Power Baseline | RL Policy (Q-Learning) | Delta (RL - Base) |
|--------|---------------------|------------------------|-------------------|
| Avg Total Reward | 96.96 | 175.12 | **+78.16 (+80.6%)** |
| Std Reward | 10.98 | 24.01 | — |
| Min Reward | 72.00 | 103.50 | — |
| Max Reward | 126.00 | 261.00 | — |
| Avg Battery Remaining | 0.00% | 0.00% | — |
| Energy Drain/Step | 5.0% (fixed) | adaptive (2-10%) | context-aware |
| Strategy | Always medium | State-adaptive | — |
| Adapts to battery state | No | Yes | — |
| Adapts to task demand | No | Yes | — |

> Evaluated over 200 episodes, seed=99. Run `python run_all.py --load_policy policy_v1.pkl`
> to reproduce. Live numbers saved to `outputs/comparison_table.md` and `outputs/index.html`.

---

## 6. Plots

### Plot 1 — Average Reward over Episodes
`outputs/plots/reward_vs_episodes.png`

Shows the raw and smoothed (window=30) reward per episode for both agents over 200
evaluation episodes. The RL agent's reward curve sits consistently above the baseline,
demonstrating that the learned Q-table outperforms the fixed heuristic.

### Plot 2 — Battery Level over Time (Single Episode)
`outputs/plots/battery_vs_time.png`

Traces battery percentage at each timestep for one representative episode (seed=123).
The RL agent conserves battery on low-demand steps (shallow slope) and expends more
on high-demand steps; the baseline drains at a constant rate (steep, straight line).

---

## 7. Results Analysis

### When RL Performs Better

- **Mixed workloads:** On low-demand tasks the RL agent switches to low power,
  conserving battery at the cost of only 2% drain instead of 5%. On high-demand
  tasks it escalates to high power when charge is sufficient, scoring 10 points
  instead of the baseline's 4. The net effect is consistently higher reward.

- **Low-battery situations:** The RL agent learns to use low power when battery
  is in the "low" zone (< 33%), squeezing more useful decision steps before
  depletion. The baseline continues at medium power, burning charge regardless.

### When RL Behaves Unexpectedly or Poorly

- **Early training (high ε):** The first ~100 episodes produce very low rewards as
  the agent explores randomly. This is expected but visible as high variance at the
  start of the reward curve.

- **Workload spikes:** If all tasks are high-demand in a single episode, the RL
  agent may run out of battery faster than expected because its Q-values were
  learned under a uniform task distribution.

- **Rare state (low battery + high demand):** This state is visited infrequently
  during training, so the Q-value estimate is noisy. The agent may make a suboptimal
  choice (e.g., still picking high power when battery is critically low).

### Sensitivity to Traffic Pattern Changes

If the task distribution shifts from uniform to predominantly high-demand (analogous
to sudden traffic surges on one lane), the RL policy's advantage shrinks because:
1. High-demand steps consume 10% battery each, depleting charge faster.
2. The Q-table was not trained on this imbalanced distribution.

Conversely, if demand is predominantly low, the RL policy significantly outperforms
the baseline because it selects low power far more often, saving ~3% per step.

---

## 8. SDG Impact

### SDG 7 — Affordable and Clean Energy

> "The RL agent achieves 80.6% higher average reward than the fixed-power baseline
> (175 vs 97), meaning it completes far more tasks per battery charge. This directly
> supports SDG 7 by reducing electricity consumption per device and extending battery
> life across IoT infrastructure at scale."

The RL agent's ability to match power to demand means energy is not wasted on
low-priority tasks. Scaled across thousands of edge devices in a smart city, even a
15–20% improvement in energy efficiency meaningfully reduces electricity demand and
operating costs, making sustainable IoT deployment more affordable.

### SDG 11 — Sustainable Cities and Communities

> "Adaptive power management reduces the resource footprint of smart-city IoT
> deployments, enabling denser sensor networks without proportionally increasing
> power demand — supporting SDG 11's goal of inclusive, safe, and sustainable cities."

Smart traffic sensors, air quality monitors, and environmental IoT nodes are core
infrastructure for sustainable cities. If each node wastes 20% of its battery on
fixed-power decisions, the infrastructure requires more frequent battery replacement,
more charging stations, and higher grid load. RL-based adaptive management reduces
all of these costs, directly supporting sustainable urban infrastructure.

---

## 9. Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Tabular Q-learning | Does not scale to continuous or high-dimensional states | Use DQN / PPO for richer environments |
| Uniform task distribution | Policy degrades on real bursty workloads | Monitor KL-divergence; retrain on real data |
| Static policy | No online adaptation after deployment | Add online Q-table updates or periodic re-training |
| Deterministic drain | Real hardware has noise and temperature variance | Add stochastic drain to simulator |
| No multi-device coordination | Optimises each device independently | Multi-agent RL for fleet-level optimisation |

---

## 10. Deployment Instructions

### Option A — Docker (Recommended, one command)

```bash
git clone <your-github-repo-url>
cd rl_final
docker-compose up
# Outputs appear in ./outputs/
# Open outputs/index.html in a browser for the full report
```

### Option B — Docker without Compose

```bash
docker build -t rl-energy .
docker run --rm -v "$(pwd)/outputs:/app/outputs" rl-energy
```

### Option C — Local Python

```bash
pip install -r requirements.txt

# Full pipeline (train + compare + report):
python run_all.py

# Skip training, use saved policy:
python run_all.py --load_policy policy_v1.pkl

# Individual steps:
python train.py --config configs/qlearning_v1.yaml
python train.py --config configs/qlearning_v2.yaml
python compare.py --load_policy policy_v1.pkl --episodes 200
python baseline.py --episodes 200
```

### Reproducing a specific experiment by Git tag

```bash
git checkout exp-qlearning-1   # restore exact state of v1 run
python train.py --config configs/qlearning_v1.yaml --run_id run_1
git checkout main
```

---

## 11. Repository Structure

```
rl_final/
├── environment.py          ← Battery-task simulator (RL environment)
├── agent.py                ← Tabular Q-Learning agent
├── train.py                ← MLOps-enhanced training (YAML config, CSV + JSON logging)
├── baseline.py             ← Fixed medium-power baseline agent
├── compare.py              ← Head-to-head evaluation + matplotlib plots
├── run_all.py              ← Full pipeline entry point (Docker CMD)
├── utils.py                ← Shared helpers
├── configs/
│   ├── qlearning_v1.yaml   ← exp-qlearning-1: 500 eps, lr=0.1, γ=0.95
│   └── qlearning_v2.yaml   ← exp-qlearning-2: 1000 eps, lr=0.15, γ=0.99
├── Dockerfile              ← Containerised runner
├── docker-compose.yml      ← One-command deployment
├── requirements.txt        ← numpy, matplotlib, pyyaml
├── results.csv             ← Experiment tracking (auto-generated)
├── logs/                   ← Per-episode JSON logs
├── outputs/                ← Generated by run_all.py (plots + HTML report)
│   ├── plots/
│   │   ├── reward_vs_episodes.png
│   │   └── battery_vs_time.png
│   ├── comparison_table.md
│   └── index.html          ← Self-contained HTML report (open in browser)
├── policy_v1.pkl           ← Trained Q-table (exp-qlearning-1)
└── policy_v2.pkl           ← Trained Q-table (exp-qlearning-2)
```
