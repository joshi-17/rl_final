"""
compare.py — Baseline vs RL Policy Evaluation & Visualisation (Part 3)

Steps:
  1. Train (or load) an RL agent using the Q-learning policy.
  2. Run the fixed baseline agent for the same number of episodes.
  3. Compare average reward and average battery remaining.
  4. Plot:
       a) Reward vs episodes   (both agents, smoothed)
       b) Battery level vs time (single representative episode)
  5. Print a comparison table.

Usage:
    python compare.py                          # trains fresh RL agent
    python compare.py --load_policy policy_v1.pkl   # loads existing policy
    python compare.py --episodes 300
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from environment import EnergyEnvironment
from agent import QLearningAgent
from baseline import BaselineAgent, run_baseline
from utils import moving_average, safe_mkdir


# -- Paths ---------------------------------------------------------------------
PLOTS_DIR   = "plots"
POLICY_PATH = "policy_v1.pkl"


# -----------------------------------------------------------------------------
def train_rl_agent(episodes=500, seed=42):
    """Train a Q-learning agent from scratch and return it."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    env   = EnergyEnvironment(seed=seed)
    agent = QLearningAgent(lr=0.1, gamma=0.95, epsilon=1.0,
                           epsilon_min=0.01, epsilon_decay=0.995)

    print(f"  Training RL agent for {episodes} episodes...")
    for ep in range(1, episodes + 1):
        state        = env.reset()
        while not env.done:
            action                      = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()

        if ep % 100 == 0:
            print(f"    ep={ep}/{episodes}  ε={agent.epsilon:.4f}")

    agent.epsilon = 0.0   # switch to pure exploitation for evaluation
    print("  RL training done.\n")
    return agent


def load_rl_agent(policy_path):
    """Load a previously trained Q-learning agent from a pickle file."""
    agent = QLearningAgent()
    agent.load(policy_path)
    agent.epsilon = 0.0   # pure exploitation during evaluation
    return agent


# -----------------------------------------------------------------------------
def run_rl_eval(agent, episodes=200, seed=99):
    """
    Evaluate a trained RL agent (epsilon=0, greedy) over multiple episodes.

    Returns
    -------
    episode_rewards : list[float]
    episode_battery : list[float]
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    env = EnergyEnvironment(seed=seed)
    episode_rewards = []
    episode_battery = []

    for ep in range(1, episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        while not env.done:
            action                      = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state        = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        episode_battery.append(env.get_battery_percentage())

        if ep % 50 == 0:
            avg_r = float(np.mean(episode_rewards[-50:]))
            print(f"  [RL Agent]  ep={ep:>4}/{episodes}  avg_reward={avg_r:>8.2f}  "
                  f"battery={env.get_battery_percentage():.1f}%")

    return episode_rewards, episode_battery


# -----------------------------------------------------------------------------
def record_episode_trace(agent_or_baseline, seed=123):
    """
    Run ONE episode and record battery % at every step.
    Used for the "battery vs time" plot.
    """
    import random
    random.seed(seed)
    env = EnergyEnvironment(seed=seed)
    state = env.reset()

    battery_trace = [env.get_battery_percentage()]
    while not env.done:
        action         = agent_or_baseline.choose_action(state)
        state, _, _, _ = env.step(action)
        battery_trace.append(env.get_battery_percentage())

    return battery_trace


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_reward_vs_episodes(rl_rewards, bl_rewards, output_path, window=30):
    """
    Plot smoothed episode rewards for both RL agent and baseline.
    """
    rl_smooth = moving_average(rl_rewards, window=window)
    bl_smooth = moving_average(bl_rewards, window=window)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Raw reward (faint)
    ax.plot(rl_rewards, color="#4C9BE8", alpha=0.2, linewidth=0.8, label="_nolegend_")
    ax.plot(bl_rewards, color="#F4845F", alpha=0.2, linewidth=0.8, label="_nolegend_")

    # Smoothed reward
    ax.plot(rl_smooth, color="#1A6BB5", linewidth=2.0, label=f"RL Agent (Q-Learning, smoothed w={window})")
    ax.plot(bl_smooth, color="#C94A2A", linewidth=2.0, label=f"Baseline (fixed medium power, smoothed)")

    ax.set_title("Reward vs Episodes — RL Agent vs Baseline", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotation: final averages
    ax.axhline(np.mean(rl_rewards[-50:]), color="#1A6BB5", linestyle="--", alpha=0.5)
    ax.axhline(np.mean(bl_rewards[-50:]), color="#C94A2A", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {output_path}")


def plot_battery_vs_time(rl_trace, bl_trace, output_path):
    """
    Plot battery level over time for one representative episode.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rl_trace, color="#1A6BB5", linewidth=2.2, marker="o",
            markersize=3, label="RL Agent (adaptive power)")
    ax.plot(bl_trace, color="#C94A2A", linewidth=2.2, marker="s",
            markersize=3, label="Baseline (always medium power)")

    # Shade battery zones
    ax.axhspan(0,  33, alpha=0.07, color="red")
    ax.axhspan(33, 66, alpha=0.07, color="orange")
    ax.axhspan(66, 100, alpha=0.07, color="green")

    ax.text(0.5, 15, "Low battery zone",  color="red",    alpha=0.5, fontsize=9)
    ax.text(0.5, 48, "Medium battery zone", color="darkorange", alpha=0.5, fontsize=9)
    ax.text(0.5, 80, "High battery zone", color="green",  alpha=0.5, fontsize=9)

    ax.set_title("Battery Level vs Time — Single Episode Trace", fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep (within episode)", fontsize=12)
    ax.set_ylabel("Battery Remaining (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] Saved -> {output_path}")


# -----------------------------------------------------------------------------
def print_comparison_table(rl_rewards, bl_rewards, rl_battery, bl_battery):
    """Print a formatted comparison table to the console."""
    rl_avg_r = np.mean(rl_rewards)
    bl_avg_r = np.mean(bl_rewards)
    rl_avg_b = np.mean(rl_battery)
    bl_avg_b = np.mean(bl_battery)

    reward_diff = rl_avg_r - bl_avg_r
    battery_diff = rl_avg_b - bl_avg_b

    print("\n" + "=" * 62)
    print("  COMPARISON TABLE: Baseline vs RL Policy")
    print("=" * 62)
    print(f"  {'Metric':<28} {'Baseline':>10}  {'RL Policy':>10}  {'Delta (RL-Base)':>11}")
    print("-" * 62)
    print(f"  {'Avg reward':<28} {bl_avg_r:>10.2f}  {rl_avg_r:>10.2f}  {reward_diff:>+11.2f}")
    print(f"  {'Std reward':<28} {np.std(bl_rewards):>10.2f}  {np.std(rl_rewards):>10.2f}  {'':>11}")
    print(f"  {'Avg battery remaining (%)':<28} {bl_avg_b:>10.2f}  {rl_avg_b:>10.2f}  {battery_diff:>+11.2f}")
    print(f"  {'Min reward':<28} {np.min(bl_rewards):>10.2f}  {np.min(rl_rewards):>10.2f}  {'':>11}")
    print(f"  {'Max reward':<28} {np.max(bl_rewards):>10.2f}  {np.max(rl_rewards):>10.2f}  {'':>11}")
    print("=" * 62)
    winner = "RL Policy" if rl_avg_r > bl_avg_r else "Baseline"
    print(f"\n  > Higher avg reward: {winner}")
    if rl_avg_r > bl_avg_r:
        pct = 100.0 * reward_diff / abs(bl_avg_r) if bl_avg_r != 0 else float("inf")
        print(f"  > RL improves over baseline by {pct:.1f}%")
    print("=" * 62 + "\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RL agent on Smart Energy Management"
    )
    parser.add_argument("--episodes",    type=int,  default=200,
                        help="Evaluation episodes for both agents (default 200)")
    parser.add_argument("--train_ep",   type=int,  default=500,
                        help="RL training episodes if training from scratch (default 500)")
    parser.add_argument("--load_policy", type=str, default=None,
                        help="Path to existing pkl policy (skip training if provided)")
    parser.add_argument("--seed",        type=int,  default=42)
    args = parser.parse_args()

    safe_mkdir(PLOTS_DIR)

    # -- 1. Get RL agent -------------------------------------------------------
    if args.load_policy and os.path.isfile(args.load_policy):
        print(f"\n  Loading RL policy from {args.load_policy} ...")
        rl_agent = load_rl_agent(args.load_policy)
    else:
        print("\n  No saved policy found — training RL agent from scratch ...")
        rl_agent = train_rl_agent(episodes=args.train_ep, seed=args.seed)
        rl_agent.save(POLICY_PATH)

    # -- 2. Evaluate RL agent --------------------------------------------------
    print(f"\n  Evaluating RL agent for {args.episodes} episodes...")
    rl_rewards, rl_battery = run_rl_eval(rl_agent, episodes=args.episodes, seed=99)

    # -- 3. Evaluate baseline --------------------------------------------------
    print(f"\n  Evaluating Baseline agent for {args.episodes} episodes...")
    bl_agent  = BaselineAgent()
    bl_rewards, bl_battery, _ = run_baseline(
        episodes=args.episodes, seed=99, verbose=True
    )

    # -- 4. Comparison table ---------------------------------------------------
    print_comparison_table(rl_rewards, bl_rewards, rl_battery, bl_battery)

    # -- 5. Battery episode traces (single episode) ----------------------------
    rl_trace = record_episode_trace(rl_agent, seed=123)
    bl_trace = record_episode_trace(bl_agent,  seed=123)

    # -- 6. Plots --------------------------------------------------------------
    plot_reward_vs_episodes(
        rl_rewards, bl_rewards,
        output_path=os.path.join(PLOTS_DIR, "reward_vs_episodes.png"),
    )
    plot_battery_vs_time(
        rl_trace, bl_trace,
        output_path=os.path.join(PLOTS_DIR, "battery_vs_time.png"),
    )

    print(f"  ✓ Plots saved to ./{PLOTS_DIR}/")
    print(f"  ✓ Comparison complete.\n")
