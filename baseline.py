"""
baseline.py — Fixed-Policy Baseline for Smart Energy Management

The baseline agent always chooses medium power (action=1) regardless of
battery level or task demand. This represents a naive, uninformed strategy
that a device might use by default.

It serves as a reference point: if the RL agent cannot outperform this
simple heuristic, the training has failed.

Usage:
    python baseline.py
    python baseline.py --episodes 300
"""

import argparse
import numpy as np

from environment import EnergyEnvironment
from utils import action_to_name, battery_level_to_name, task_demand_to_name


# -----------------------------------------------------------------------------
class BaselineAgent:
    """
    Fixed-policy agent that always selects medium power (action=1).

    This is the simplest possible policy:
      - It never adapts to battery state.
      - It never adapts to task demand.
      - It represents the "do nothing smart" default.
    """

    FIXED_ACTION = 1  # medium_power

    def choose_action(self, state):
        """Ignore state; always return medium_power."""
        return self.FIXED_ACTION

    @property
    def name(self):
        return f"Baseline (always {action_to_name(self.FIXED_ACTION)})"


# -----------------------------------------------------------------------------
def run_baseline(episodes=200, seed=42, verbose=True):
    """
    Evaluate the baseline agent over a number of episodes.

    Parameters
    ----------
    episodes : int   Number of evaluation episodes
    seed     : int   RNG seed
    verbose  : bool  Print episode summaries

    Returns
    -------
    episode_rewards  : list[float]
    episode_battery  : list[float]   remaining battery % at episode end
    episode_steps    : list[int]     steps taken per episode
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    env    = EnergyEnvironment(seed=seed)
    agent  = BaselineAgent()

    episode_rewards = []
    episode_battery = []
    episode_steps   = []

    for ep in range(1, episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        steps        = 0

        while not env.done:
            action                      = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            state        = next_state
            total_reward += reward
            steps        += 1

        episode_rewards.append(total_reward)
        episode_battery.append(env.get_battery_percentage())
        episode_steps.append(steps)

        if verbose and ep % 50 == 0:
            avg_r = float(np.mean(episode_rewards[-50:]))
            print(f"  [Baseline] ep={ep:>4}/{episodes}  avg_reward={avg_r:>8.2f}  "
                  f"battery={env.get_battery_percentage():.1f}%")

    return episode_rewards, episode_battery, episode_steps


# -----------------------------------------------------------------------------
def print_baseline_summary(episode_rewards, episode_battery, episode_steps):
    """Print a human-readable summary of the baseline evaluation."""
    print("\n" + "=" * 50)
    print("  BASELINE EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Episodes         : {len(episode_rewards)}")
    print(f"  Avg reward       : {np.mean(episode_rewards):.2f}")
    print(f"  Std reward       : {np.std(episode_rewards):.2f}")
    print(f"  Min reward       : {np.min(episode_rewards):.2f}")
    print(f"  Max reward       : {np.max(episode_rewards):.2f}")
    print(f"  Avg battery left : {np.mean(episode_battery):.2f}%")
    print(f"  Avg steps/ep     : {np.mean(episode_steps):.1f}")
    print("=" * 50 + "\n")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline agent evaluation")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of evaluation episodes (default 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default 42)")
    args = parser.parse_args()

    print(f"\n  Running baseline agent for {args.episodes} episodes...")
    rewards, battery, steps = run_baseline(
        episodes=args.episodes, seed=args.seed, verbose=True
    )
    print_baseline_summary(rewards, battery, steps)
