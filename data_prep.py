"""Data preparation utilities for reproducible RL experiments.

The simulator is the source of truth for this project, but a small tabular
dataset makes the data-cleaning and feature-engineering steps explicit and
auditable for the final pipeline.
"""

import argparse
import csv
import random
from pathlib import Path

from utils import safe_mkdir


RAW_PATH = Path("data/raw_tasks.csv")
CLEAN_PATH = Path("data/clean_tasks.csv")


def generate_raw_tasks(path=RAW_PATH, rows=300, seed=42):
    """Generate a deterministic task-demand dataset with a few dirty rows."""
    random.seed(seed)
    safe_mkdir(str(Path(path).parent))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task_id", "battery_pct", "task_demand"])
        writer.writeheader()
        for idx in range(rows):
            demand = random.choice(["low", "medium", "high", ""])
            battery = random.choice([random.randint(5, 100), "", 130, -5])
            writer.writerow({"task_id": idx, "battery_pct": battery, "task_demand": demand})
    return Path(path)


def _battery_bucket(battery_pct):
    if battery_pct > 66:
        return "high"
    if battery_pct > 33:
        return "medium"
    return "low"


def clean_and_engineer(raw_path=RAW_PATH, clean_path=CLEAN_PATH):
    """Clean invalid rows and add engineered features used in monitoring."""
    safe_mkdir(str(Path(clean_path).parent))
    cleaned = []
    with open(raw_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                battery = float(row["battery_pct"])
            except (TypeError, ValueError):
                continue
            demand = (row.get("task_demand") or "").strip().lower()
            if demand not in {"low", "medium", "high"}:
                continue
            battery = min(100.0, max(0.0, battery))
            demand_id = {"low": 0, "medium": 1, "high": 2}[demand]
            cleaned.append(
                {
                    "task_id": row["task_id"],
                    "battery_pct": round(battery, 2),
                    "battery_bucket": _battery_bucket(battery),
                    "task_demand": demand,
                    "task_demand_id": demand_id,
                    "is_high_demand": int(demand == "high"),
                }
            )

    with open(clean_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "task_id",
            "battery_pct",
            "battery_bucket",
            "task_demand",
            "task_demand_id",
            "is_high_demand",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)
    return Path(clean_path), len(cleaned)


def main():
    parser = argparse.ArgumentParser(description="Generate and clean task data")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = generate_raw_tasks(rows=args.rows, seed=args.seed)
    clean, n_rows = clean_and_engineer(raw)
    print(f"Generated {raw}")
    print(f"Cleaned {n_rows} rows -> {clean}")


if __name__ == "__main__":
    main()
