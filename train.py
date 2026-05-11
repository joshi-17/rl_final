"""
train.py — MLOps-enhanced training script

Extends the core agent with:
  - YAML config loading via --config argument
  - Experiment tracking: results appended to results.csv
  - Per-episode reward/drain/task-score logging: logs/run_<id>.json
  - Versioned run IDs  (experiment tags: exp-qlearning-1, exp-qlearning-2)

Usage:
    python train.py --config configs/qlearning_v1.yaml
    python train.py --config configs/qlearning_v2.yaml --run_id run_2
"""

import argparse
import csv
import json
import os
import time
import numpy as np

# ── Try to import yaml; fall back to a minimal parser if unavailable ──────────
try:
    import yaml
    def load_yaml(path):
        with open(path, "r") as fh:
            return yaml.safe_load(fh)
except ImportError:
    # Minimal YAML parser for simple key:value files (no external deps needed)
    def load_yaml(path):
        """Minimal YAML loader for flat/nested key:value files."""
        def _parse_value(v):
            v = v.strip()
            if v.lower() == "true":  return True
            if v.lower() == "false": return False
            try: return int(v)
            except ValueError: pass
            try: return float(v)
            except ValueError: pass
            return v.strip('"').strip("'")

        result, stack = {}, [(0, result)]
        with open(path) as fh:
            for raw in fh:
                line = raw.rstrip()
                if not line or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                key_part = line.lstrip()
                if ":" not in key_part:
                    continue
                key, _, val = key_part.partition(":")
                key = key.strip()
                val = val.strip()
                # Pop stack to correct indent level
                while len(stack) > 1 and stack[-1][0] >= indent:
                    stack.pop()
                parent = stack[-1][1]
                if val:
                    parent[key] = _parse_value(val)
                else:
                    parent[key] = {}
                    stack.append((indent + 1, parent[key]))
        return result

# ── Import core modules (reused from Part 1) ──────────────────────────────────
from environment import EnergyEnvironment
from agent import QLearningAgent
from utils import moving_average, print_episode_log, safe_mkdir


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Q-learning agent for Smart Energy Management"
    )
    parser.add_argument(
        "--config", type=str, default="configs/qlearning_v1.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--run_id", type=str, default=None,
        help="Override run ID (default: auto-generated from timestamp)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path):
    """Load YAML config and return a flat settings dict."""
    raw = load_yaml(config_path)

    # Flatten nested YAML into easy-access dict
    cfg = {
        # experiment metadata
        "exp_name"     : raw.get("experiment", {}).get("name", "exp-energy-1"),
        "exp_version"  : raw.get("experiment", {}).get("version", "v1"),
        # training
        "episodes"     : raw.get("training", {}).get("episodes", 500),
        "log_interval" : raw.get("training", {}).get("log_interval", 100),
        "seed"         : raw.get("training", {}).get("seed", 42),
        # hyper-params
        "lr"           : raw.get("hyperparameters", {}).get("learning_rate", 0.1),
        "gamma"        : raw.get("hyperparameters", {}).get("gamma", 0.95),
        "epsilon"      : raw.get("hyperparameters", {}).get("epsilon", 1.0),
        "epsilon_min"  : raw.get("hyperparameters", {}).get("epsilon_min", 0.01),
        "epsilon_decay": raw.get("hyperparameters", {}).get("epsilon_decay", 0.995),
        # environment
        "max_battery"  : raw.get("environment", {}).get("max_battery", 100),
        # output
        "policy_path"  : raw.get("output", {}).get("policy_path", "policy_v1.pkl"),
        "results_csv"  : raw.get("output", {}).get("results_csv", "results.csv"),
        "logs_dir"     : raw.get("output", {}).get("logs_dir", "logs"),
    }
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
def append_results_csv(filepath, row: dict):
    """
    Append one result row to results.csv (create with header if missing).

    Columns: run_id, episodes, avg_reward, avg_battery_remaining,
             avg_drain_per_step, avg_task_score, epsilon, learning_rate
    """
    fieldnames = [
        "run_id", "episodes", "avg_reward",
        "avg_battery_remaining", "avg_drain_per_step",
        "avg_task_score", "epsilon", "learning_rate",
    ]
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [mlops] Results appended to {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
def save_episode_log(logs_dir, run_id, episode_rewards, episode_battery,
                     episode_drains, episode_task_scores):
    """
    Save per-episode metrics to logs/<run_id>.json.

    Experiment tags: exp-qlearning-1  exp-qlearning-2
    Metrics logged per episode:
      - rewards          : total reward
      - battery_remaining: % battery left at episode end
      - drain_per_step   : avg battery % drained per timestep (energy efficiency)
      - task_score       : avg task completion score per timestep
    """
    safe_mkdir(logs_dir)
    log_path = os.path.join(logs_dir, f"{run_id}.json")
    payload = {
        "run_id"             : run_id,
        "experiment_tags"    : ["exp-qlearning-1", "exp-qlearning-2"],
        "n_episodes"         : len(episode_rewards),
        "episode_rewards"    : [round(r, 4) for r in episode_rewards],
        "episode_battery"    : [round(b, 2) for b in episode_battery],
        "episode_drain_per_step"  : [round(d, 4) for d in episode_drains],
        "episode_task_scores"     : [round(s, 4) for s in episode_task_scores],
    }
    with open(log_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [mlops] Episode log saved to {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict, run_id: str):
    """
    Main training loop — reads all settings from cfg dict.
    Tracks metrics and writes logs + results CSV.
    """
    np.random.seed(cfg["seed"])

    # ── Experiment: exp-qlearning-1 (standard run) ────────────────────────────
    # ── Experiment: exp-qlearning-2 (tuned exploration run) ──────────────────
    print(f"\n{'═'*55}")
    print(f"  Experiment : {cfg['exp_name']}  ({cfg['exp_version']})")
    print(f"  Run ID     : {run_id}")
    print(f"  Config     : episodes={cfg['episodes']}  lr={cfg['lr']}  "
          f"γ={cfg['gamma']}  ε={cfg['epsilon']}")
    print(f"{'═'*55}")

    env = EnergyEnvironment(max_battery=cfg["max_battery"], seed=cfg["seed"])
    agent = QLearningAgent(
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
    )

    episode_rewards     = []
    episode_battery     = []
    episode_drains      = []   # avg battery % drained per step (energy efficiency)
    episode_task_scores = []   # avg task completion score per step

    for ep in range(1, cfg["episodes"] + 1):
        state        = env.reset()
        total_reward = 0.0
        step_drains  = []
        step_scores  = []

        while not env.done:
            action                         = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward
            # battery_penalty = drain * 0.5  →  drain = battery_penalty * 2
            step_drains.append(info["battery_penalty"] * 2)
            step_scores.append(info["task_score"])

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_battery.append(env.get_battery_percentage())
        episode_drains.append(float(np.mean(step_drains)) if step_drains else 0.0)
        episode_task_scores.append(float(np.mean(step_scores)) if step_scores else 0.0)

        if ep % cfg["log_interval"] == 0:
            avg_r = float(np.mean(episode_rewards[-cfg["log_interval"]:]))
            print_episode_log(ep, cfg["episodes"], avg_r, agent.epsilon)

    # ── Save trained policy ───────────────────────────────────────────────────
    agent.save(cfg["policy_path"])

    # ── Compute summary metrics ───────────────────────────────────────────────
    window          = min(100, cfg["episodes"])
    avg_reward      = float(np.mean(episode_rewards[-window:]))
    avg_battery     = float(np.mean(episode_battery[-window:]))
    avg_drain       = float(np.mean(episode_drains[-window:]))
    avg_task_score  = float(np.mean(episode_task_scores[-window:]))

    print(f"\n  Final {window}-ep avg reward       : {avg_reward:.2f}")
    print(f"  Final {window}-ep avg battery      : {avg_battery:.2f}%")
    print(f"  Final {window}-ep avg drain/step   : {avg_drain:.2f}%")
    print(f"  Final {window}-ep avg task score   : {avg_task_score:.2f}")

    # ── MLOps: append to results.csv ──────────────────────────────────────────
    result_row = {
        "run_id"               : run_id,
        "episodes"             : cfg["episodes"],
        "avg_reward"           : round(avg_reward, 4),
        "avg_battery_remaining": round(avg_battery, 4),
        "avg_drain_per_step"   : round(avg_drain, 4),
        "avg_task_score"       : round(avg_task_score, 4),
        "epsilon"              : round(cfg["epsilon"], 4),
        "learning_rate"        : cfg["lr"],
    }
    append_results_csv(cfg["results_csv"], result_row)

    # ── MLOps: save per-episode log ───────────────────────────────────────────
    save_episode_log(
        cfg["logs_dir"], run_id,
        episode_rewards, episode_battery,
        episode_drains, episode_task_scores,
    )

    return agent, episode_rewards, episode_battery


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # Load config
    cfg = load_config(args.config)

    # Determine run ID
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"run_{int(time.time())}"

    # Run training
    agent, rewards, battery = train(cfg, run_id)

    print(f"\n  ✓ Training complete. Run ID: {run_id}")
    print(f"  Policy saved to: {cfg['policy_path']}")
    print(f"  Results CSV    : {cfg['results_csv']}")
    print(f"  Episode log    : {cfg['logs_dir']}/{run_id}.json")
