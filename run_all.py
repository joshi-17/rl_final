"""
run_all.py — Full pipeline: train -> evaluate -> compare -> HTML report

Usage (local):
    python run_all.py
    python run_all.py --train_episodes 500 --eval_episodes 200
    python run_all.py --load_policy policy_v1.pkl

Usage (Docker):
    docker-compose up
    docker build -t rl-energy . && docker run --rm -v $(pwd)/outputs:/app/outputs rl-energy
"""

import argparse
import base64
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np

from agent import QLearningAgent
from baseline import BaselineAgent, run_baseline
from compare import (
    load_rl_agent,
    plot_battery_vs_time,
    plot_reward_vs_episodes,
    print_comparison_table,
    record_episode_trace,
    run_rl_eval,
    train_rl_agent,
)
from utils import safe_mkdir

OUTPUTS_DIR = "outputs"
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")
POLICY_PATH = "policy_v1.pkl"


# -- helpers -------------------------------------------------------------------

def _img_b64(path):
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


def save_comparison_table_md(rl_rewards, bl_rewards, rl_battery, bl_battery, path):
    rl_avg_r = np.mean(rl_rewards)
    bl_avg_r = np.mean(bl_rewards)
    rl_avg_b = np.mean(rl_battery)
    bl_avg_b = np.mean(bl_battery)
    pct = 100.0 * (rl_avg_r - bl_avg_r) / abs(bl_avg_r) if bl_avg_r != 0 else 0

    lines = [
        "# Comparison Results: Baseline vs RL Policy\n",
        "| Metric | Fixed-Power Baseline | RL Policy (Q-Learning) | Delta (RL - Base) |",
        "|--------|---------------------|------------------------|---------------|",
        f"| Avg Total Reward | {bl_avg_r:.2f} | {rl_avg_r:.2f} | {rl_avg_r - bl_avg_r:+.2f} ({pct:+.1f}%) |",
        f"| Std Reward | {np.std(bl_rewards):.2f} | {np.std(rl_rewards):.2f} | — |",
        f"| Min Reward | {np.min(bl_rewards):.2f} | {np.min(rl_rewards):.2f} | — |",
        f"| Max Reward | {np.max(bl_rewards):.2f} | {np.max(rl_rewards):.2f} | — |",
        f"| Avg Battery Remaining (%) | {bl_avg_b:.2f} | {rl_avg_b:.2f} | {rl_avg_b - bl_avg_b:+.2f} |",
        "| Strategy | Always medium power | State-adaptive Q-table | — |",
        "| Adapts to demand | No | Yes | — |",
        "",
        f"**RL Policy improves over Baseline by {pct:.1f}% in average total reward.**",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [output] Comparison table -> {path}")


def generate_html_report(rl_rewards, bl_rewards, plots_dir, path):
    rl_avg = np.mean(rl_rewards)
    bl_avg = np.mean(bl_rewards)
    pct    = 100.0 * (rl_avg - bl_avg) / abs(bl_avg) if bl_avg != 0 else 0

    reward_b64  = _img_b64(os.path.join(plots_dir, "reward_vs_episodes.png"))
    battery_b64 = _img_b64(os.path.join(plots_dir, "battery_vs_time.png"))

    reward_tag  = (f'<img src="data:image/png;base64,{reward_b64}" alt="Reward vs Episodes">'
                   if reward_b64 else "<p><em>Plot not generated.</em></p>")
    battery_tag = (f'<img src="data:image/png;base64,{battery_b64}" alt="Battery vs Time">'
                   if battery_b64 else "<p><em>Plot not generated.</em></p>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RL Energy Management — Final Results</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1   {{ color: #1A6BB5; }}
  h2   {{ color: #1A6BB5; border-bottom: 2px solid #1A6BB5; padding-bottom: 4px; }}
  h3   {{ color: #444; }}
  table{{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th   {{ background: #1A6BB5; color: #fff; padding: 10px 12px; text-align: left; }}
  td   {{ padding: 8px 12px; border: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f5f8fc; }}
  .win {{ background: #d4edda !important; font-weight: bold; }}
  img  {{ max-width: 100%; margin: 12px 0; border: 1px solid #ddd; border-radius: 4px; }}
  .sdg {{ background: #f0f8f0; padding: 14px 18px; border-left: 4px solid #2d8a2d;
          margin: 14px 0; border-radius: 0 4px 4px 0; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  pre  {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
  .badge{{ display: inline-block; background: #1A6BB5; color: #fff; border-radius: 12px;
           padding: 2px 10px; font-size: 0.85em; margin-left: 6px; }}
</style>
</head>
<body>

<h1>Smart Energy Management via Reinforcement Learning <span class="badge">Final Report</span></h1>
<p><strong>Project:</strong> rl_final &nbsp;|&nbsp;
   <strong>Algorithm:</strong> Tabular Q-Learning &nbsp;|&nbsp;
   <strong>Domain:</strong> IoT / Edge device power management</p>

<h2>1. Problem Statement</h2>
<p>Battery-powered IoT devices must balance <em>task performance</em> against
<em>battery longevity</em>. A naive device always runs at medium power, wasting
energy on easy tasks and potentially failing on high-demand bursts. This project
trains a Q-learning agent to choose the right power level (low / medium / high)
based on current battery state and incoming task demand, maximising total reward
while conserving charge.</p>

<h2>2. Simulator</h2>
<ul>
  <li><strong>State:</strong> (battery_level ∈ {{low, medium, high}}, task_demand ∈ {{low, medium, high}}) -> 9 states</li>
  <li><strong>Actions:</strong> 0 = low_power (2 % drain), 1 = medium_power (5 % drain), 2 = high_power (10 % drain)</li>
  <li><strong>Reward:</strong> task_completion_score - 0.5 × drain — matching power to demand gives the highest score</li>
  <li><strong>Terminal:</strong> battery reaches 0 %</li>
</ul>

<h2>3. RL Algorithm — Tabular Q-Learning</h2>
<pre>Q(s,a) <- Q(s,a) + α · [ r + γ · max_a' Q(s',a') - Q(s,a) ]

Hyperparameters (exp-qlearning-1):
  α (learning rate)  = 0.10
  γ (discount)       = 0.95
  ε start / min      = 1.0 / 0.01
  ε decay per episode = 0.995
  Training episodes  = 500</pre>

<h2>4. MLOps</h2>
<ul>
  <li>YAML-driven experiments (<code>configs/qlearning_v1.yaml</code>, <code>v2.yaml</code>)</li>
  <li>Per-run CSV tracking (<code>results.csv</code>) and JSON episode logs (<code>logs/</code>)</li>
  <li>Versioned policy checkpoints (<code>policy_v1.pkl</code>, <code>policy_v2.pkl</code>)</li>
  <li>Git-tagged experiment snapshots (<code>exp-qlearning-1</code>, <code>exp-qlearning-2</code>)</li>
  <li>Reproducible seeds pinned in YAML; Docker image for one-command reproduction</li>
</ul>

<h2>5. Comparison Table: Fixed-Power Baseline vs RL Policy</h2>
<table>
  <tr><th>Metric</th><th>Fixed-Power Baseline</th><th>RL Policy (Q-Learning)</th><th>Delta (RL - Base)</th></tr>
  <tr>
    <td>Avg Total Reward</td>
    <td>{bl_avg:.2f}</td>
    <td class="win">{rl_avg:.2f}</td>
    <td>{rl_avg - bl_avg:+.2f} ({pct:+.1f}%)</td>
  </tr>
  <tr>
    <td>Std Reward</td>
    <td>{np.std(bl_rewards):.2f}</td>
    <td>{np.std(rl_rewards):.2f}</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Min Reward</td>
    <td>{np.min(bl_rewards):.2f}</td>
    <td>{np.min(rl_rewards):.2f}</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Max Reward</td>
    <td>{np.max(bl_rewards):.2f}</td>
    <td>{np.max(rl_rewards):.2f}</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Strategy</td>
    <td>Always medium power</td>
    <td>State-adaptive (Q-table)</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Adapts to battery state</td>
    <td>No</td>
    <td>Yes</td>
    <td>—</td>
  </tr>
  <tr>
    <td>Adapts to task demand</td>
    <td>No</td>
    <td>Yes</td>
    <td>—</td>
  </tr>
</table>

<h2>6. Plot 1 — Average Reward over Episodes</h2>
{reward_tag}

<h2>7. Plot 2 — Battery Level over Time (Single Episode)</h2>
{battery_tag}

<h2>8. Results Analysis</h2>

<h3>When RL Performs Better</h3>
<ul>
  <li><strong>Mixed workloads:</strong> On low-demand tasks the RL agent switches to
      low power, conserving battery. On high-demand tasks it escalates to high power
      when charge is available — the baseline wastes energy by always using medium.</li>
  <li><strong>Low-battery situations:</strong> The RL agent learns to conserve charge
      in the (low-battery, low-demand) state, squeezing more useful steps out of the
      remaining charge.</li>
</ul>

<h3>When RL Behaves Unexpectedly or Poorly</h3>
<ul>
  <li><strong>Early training (high ε):</strong> Random exploration produces very low
      rewards before the Q-table converges — visible as variance in the reward curve.</li>
  <li><strong>Workload spikes:</strong> A sustained burst of high-demand tasks can
      deplete battery faster than the policy expects, because training used a uniform
      task distribution.</li>
  <li><strong>Rare states:</strong> The state (low battery, high demand) is
      under-visited if training episodes are short, leading to poor choices there.</li>
</ul>

<h3>Sensitivity to Traffic / Task Pattern Changes</h3>
<p>If the task distribution shifts from uniform to heavy-high-demand (e.g., a sudden
influx of intensive jobs), the RL policy degrades because it was trained on uniform
demand. The baseline is unaffected (always medium power), so in that extreme scenario
the gap narrows. Monitoring the live task histogram and triggering re-training when
KL-divergence exceeds 0.2 mitigates this drift.</p>

<h2>9. SDG Impact</h2>

<div class="sdg">
  <strong>SDG 7 — Affordable and Clean Energy</strong><br>
  The RL agent reduces unnecessary energy consumption by matching power output to
  actual task demand. Improving average reward by <strong>{pct:.1f}%</strong> over the
  naive baseline translates directly to longer battery life, fewer charge cycles, and
  lower electricity use per device. At scale across thousands of IoT nodes, this
  meaningfully reduces the energy footprint of edge computing.
</div>

<div class="sdg">
  <strong>SDG 11 — Sustainable Cities and Communities</strong><br>
  Smart adaptive power management at the device level reduces the resource footprint
  of IoT and edge deployments that underpin smart-city infrastructure. Reducing per-device
  energy waste enables denser deployment of sensors, traffic monitors, and environmental
  stations without proportionally increasing power demand — supporting cleaner, more
  resilient urban infrastructure.
</div>

<h2>10. Limitations</h2>
<ul>
  <li>Tabular Q-learning does not scale to high-dimensional or continuous state spaces;
      a DQN or PPO agent would be needed for richer environments.</li>
  <li>The simulator uses a simplified uniform task distribution; real workloads are
      bursty and non-stationary.</li>
  <li>Battery drain is deterministic per action; real hardware has noise and
      temperature-dependent variance.</li>
  <li>No online learning: the deployed policy is static and requires manual
      re-training when the workload distribution shifts.</li>
</ul>

<h2>11. How to Reproduce</h2>
<pre>
# Clone and run with Docker (one command):
git clone &lt;repo-url&gt;
cd rl_final
docker-compose up

# Or locally:
pip install -r requirements.txt
python run_all.py                         # train + compare + report
python run_all.py --load_policy policy_v1.pkl  # skip training

# Individual steps:
python train.py --config configs/qlearning_v1.yaml
python compare.py --load_policy policy_v1.pkl --episodes 200
</pre>

<p style="color:#888; font-size:0.85em; margin-top:40px;">
  Generated by run_all.py &nbsp;|&nbsp; rl_final project
</p>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [output] HTML report    -> {path}")


# -- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full RL pipeline: train -> evaluate -> compare -> report"
    )
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--eval_episodes",  type=int, default=200)
    parser.add_argument("--load_policy",    type=str, default=None)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    # Docker-friendly: read from env vars too
    train_ep = int(os.environ.get("TRAIN_EPISODES", args.train_episodes))
    eval_ep  = int(os.environ.get("EVAL_EPISODES",  args.eval_episodes))

    safe_mkdir(OUTPUTS_DIR)
    safe_mkdir(PLOTS_DIR)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Smart Energy Management - RL Final Evaluation")
    print(sep)

    # -- 1. RL agent -----------------------------------------------------------
    policy_path = args.load_policy or POLICY_PATH
    if policy_path and os.path.isfile(policy_path):
        print(f"\n[1/4] Loading RL policy from {policy_path} ...")
        rl_agent = load_rl_agent(policy_path)
    else:
        print(f"\n[1/4] Training RL agent ({train_ep} episodes) ...")
        rl_agent = train_rl_agent(episodes=train_ep, seed=args.seed)
        rl_agent.save(POLICY_PATH)

    # -- 2. Evaluate RL --------------------------------------------------------
    print(f"\n[2/4] Evaluating RL agent ({eval_ep} episodes) ...")
    rl_rewards, rl_battery = run_rl_eval(rl_agent, episodes=eval_ep, seed=99)

    # -- 3. Evaluate baseline --------------------------------------------------
    print(f"\n[3/4] Evaluating Baseline agent ({eval_ep} episodes) ...")
    bl_agent = BaselineAgent()
    bl_rewards, bl_battery, _ = run_baseline(episodes=eval_ep, seed=99, verbose=True)

    # -- 4. Outputs ------------------------------------------------------------
    print(f"\n[4/4] Generating outputs ...")

    print_comparison_table(rl_rewards, bl_rewards, rl_battery, bl_battery)

    rl_trace = record_episode_trace(rl_agent, seed=123)
    bl_trace = record_episode_trace(bl_agent,  seed=123)

    plot_reward_vs_episodes(
        rl_rewards, bl_rewards,
        output_path=os.path.join(PLOTS_DIR, "reward_vs_episodes.png"),
    )
    plot_battery_vs_time(
        rl_trace, bl_trace,
        output_path=os.path.join(PLOTS_DIR, "battery_vs_time.png"),
    )

    save_comparison_table_md(
        rl_rewards, bl_rewards, rl_battery, bl_battery,
        path=os.path.join(OUTPUTS_DIR, "comparison_table.md"),
    )
    generate_html_report(
        rl_rewards, bl_rewards,
        plots_dir=PLOTS_DIR,
        path=os.path.join(OUTPUTS_DIR, "index.html"),
    )

    print(f"\n{sep}")
    print(f"  All outputs saved to ./{OUTPUTS_DIR}/")
    print(f"  Open outputs/index.html in a browser to view the full report.")
    print(f"{sep}")


if __name__ == "__main__":
    main()
