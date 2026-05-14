"""Prediction logging and drift monitoring for deployed policies."""

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

from utils import safe_mkdir


PREDICTION_LOG = Path("logs/predictions.csv")
DRIFT_LOG = Path("logs/drift_metrics.jsonl")
TRAINING_TASK_DISTRIBUTION = {"low": 1 / 3, "medium": 1 / 3, "high": 1 / 3}


def log_prediction(state, action, battery_pct, task_demand, path=PREDICTION_LOG):
    safe_mkdir(str(Path(path).parent))
    file_exists = Path(path).is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        fieldnames = ["timestamp", "state", "action", "battery_pct", "task_demand"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": int(time.time()),
                "state": str(tuple(state)),
                "action": int(action),
                "battery_pct": float(battery_pct),
                "task_demand": task_demand,
            }
        )


def _kl_divergence(observed, expected):
    eps = 1e-9
    return sum(observed[k] * math.log((observed[k] + eps) / (expected[k] + eps)) for k in expected)


def compute_task_drift(events, threshold=0.2):
    """Return KL drift metrics for live task-demand events."""
    labels = list(TRAINING_TASK_DISTRIBUTION)
    counts = Counter(event["task_demand"] for event in events if event.get("task_demand") in labels)
    total = sum(counts.values())
    observed = {label: (counts[label] / total if total else 0.0) for label in labels}
    kl = _kl_divergence(observed, TRAINING_TASK_DISTRIBUTION) if total else 0.0
    return {
        "timestamp": int(time.time()),
        "window_size": total,
        "observed_distribution": observed,
        "training_distribution": TRAINING_TASK_DISTRIBUTION,
        "kl_divergence": round(kl, 6),
        "threshold": threshold,
        "drift_detected": kl > threshold,
    }


def read_prediction_events(path=PREDICTION_LOG, limit=500):
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-limit:]


def write_drift_metric(metric, path=DRIFT_LOG):
    safe_mkdir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(metric) + "\n")


def evaluate_and_log_drift(limit=500, threshold=0.2):
    metric = compute_task_drift(read_prediction_events(limit=limit), threshold=threshold)
    write_drift_metric(metric)
    return metric
