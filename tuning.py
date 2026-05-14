"""Hyperparameter tuning with seed-based cross-validation."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from agent import QLearningAgent
from compare import run_rl_eval
from environment import EnergyEnvironment
from utils import safe_mkdir


GRID = {
    "learning_rate": [0.05, 0.1, 0.15],
    "gamma": [0.9, 0.95, 0.99],
    "epsilon_decay": [0.99, 0.995, 0.998],
}


def train_candidate(params, episodes, seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    env = EnergyEnvironment(seed=seed)
    agent = QLearningAgent(
        lr=params["learning_rate"],
        gamma=params["gamma"],
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=params["epsilon_decay"],
    )
    for _ in range(episodes):
        state = env.reset()
        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        agent.decay_epsilon()
    agent.epsilon = 0.0
    return agent


def evaluate_candidate(params, train_episodes, eval_episodes, seeds):
    scores = []
    for seed in seeds:
        agent = train_candidate(params, train_episodes, seed)
        rewards, _ = run_rl_eval(agent, episodes=eval_episodes, seed=seed + 100)
        scores.append(float(np.mean(rewards)))
    return {
        **params,
        "cv_mean_reward": round(float(np.mean(scores)), 4),
        "cv_std_reward": round(float(np.std(scores)), 4),
        "fold_scores": [round(score, 4) for score in scores],
    }


def run_grid_search(train_episodes=250, eval_episodes=80, seeds=(11, 22, 33)):
    keys = list(GRID)
    rows = []
    for values in itertools.product(*(GRID[key] for key in keys)):
        params = dict(zip(keys, values))
        rows.append(evaluate_candidate(params, train_episodes, eval_episodes, seeds))
    rows.sort(key=lambda row: row["cv_mean_reward"], reverse=True)
    return rows


def write_results(rows, out_dir="outputs"):
    safe_mkdir(out_dir)
    csv_path = Path(out_dir) / "tuning_results.csv"
    json_path = Path(out_dir) / "best_hyperparameters.json"
    fieldnames = [
        "learning_rate",
        "gamma",
        "epsilon_decay",
        "cv_mean_reward",
        "cv_std_reward",
        "fold_scores",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows[0], fh, indent=2)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Run Q-learning grid search")
    parser.add_argument("--train_episodes", type=int, default=250)
    parser.add_argument("--eval_episodes", type=int, default=80)
    args = parser.parse_args()
    rows = run_grid_search(args.train_episodes, args.eval_episodes)
    csv_path, json_path = write_results(rows)
    print(f"Best: {rows[0]}")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
