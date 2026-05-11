"""
utils.py — Shared helper functions for the Smart Energy RL project.

Provides:
  - moving_average          : smooth noisy reward curves
  - print_training_summary  : compare two policy runs
  - discretise_battery      : continuous % → 0/1/2
  - action_to_name          : int → readable string
  - battery_level_to_name   : int → readable string
  - task_demand_to_name     : int → readable string
  - safe_mkdir              : create a directory if it does not exist
"""

import os
import numpy as np


# ── Label helpers ──────────────────────────────────────────────────────────────

def action_to_name(action):
    """Convert action index to a human-readable power-level name."""
    names = {0: "low_power", 1: "medium_power", 2: "high_power"}
    return names.get(action, "unknown")


def battery_level_to_name(level):
    """Convert discrete battery level (0/1/2) to a readable string."""
    names = {0: "low", 1: "medium", 2: "high"}
    return names.get(level, "unknown")


def task_demand_to_name(demand):
    """Convert task demand index (0/1/2) to a readable string."""
    names = {0: "low", 1: "medium", 2: "high"}
    return names.get(demand, "unknown")


# ── Numerical helpers ─────────────────────────────────────────────────────────

def moving_average(data, window=50):
    """
    Compute a trailing moving average of a list of numbers.

    Parameters
    ----------
    data   : list[float]
    window : int  look-back window size

    Returns
    -------
    list[float]  same length as data
    """
    if len(data) == 0:
        return []
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start: i + 1])))
    return result


def discretise_battery(battery_pct):
    """
    Map a continuous battery percentage (0-100) to a discrete level.

    Returns
    -------
    0  if battery_pct ≤ 33   (low)
    1  if 33 < battery_pct ≤ 66  (medium)
    2  if battery_pct > 66   (high)
    """
    if battery_pct > 66:
        return 2
    elif battery_pct > 33:
        return 1
    else:
        return 0


# ── Reporting helpers ─────────────────────────────────────────────────────────

def print_training_summary(rewards_v1, rewards_v2, window=100):
    """
    Print a side-by-side summary comparing two training runs.

    Parameters
    ----------
    rewards_v1 : list[float]  episode rewards from policy v1
    rewards_v2 : list[float]  episode rewards from policy v2
    window     : int          episodes used for the final average
    """
    w1 = min(window, len(rewards_v1))
    w2 = min(window, len(rewards_v2))

    avg1 = float(np.mean(rewards_v1[-w1:])) if rewards_v1 else 0.0
    avg2 = float(np.mean(rewards_v2[-w2:])) if rewards_v2 else 0.0

    print("\n" + "=" * 50)
    print("       TRAINING SUMMARY")
    print("=" * 50)
    print(f"  Policy V1 — episodes: {len(rewards_v1):>5}  "
          f"final {w1}-ep avg reward: {avg1:>8.2f}")
    print(f"  Policy V2 — episodes: {len(rewards_v2):>5}  "
          f"final {w2}-ep avg reward: {avg2:>8.2f}")
    winner = "V2" if avg2 > avg1 else "V1"
    print(f"\n  ► Best policy: {winner}")
    print("=" * 50 + "\n")


def print_episode_log(episode, total_episodes, avg_reward, epsilon, extra=""):
    """Formatted console log for one checkpoint during training."""
    bar_len   = 20
    filled    = int(bar_len * (episode / total_episodes))
    bar       = "█" * filled + "░" * (bar_len - filled)
    pct       = 100.0 * episode / total_episodes
    print(f"  [{bar}] {pct:5.1f}%  ep={episode:>5}/{total_episodes}  "
          f"avg_r={avg_reward:>8.2f}  ε={epsilon:.4f}  {extra}")


# ── File system helpers ───────────────────────────────────────────────────────

def safe_mkdir(path):
    """Create directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
