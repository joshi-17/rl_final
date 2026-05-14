"""Simple local model registry for versioned policy artifacts."""

import json
import shutil
import time
from pathlib import Path

from utils import safe_mkdir


REGISTRY_DIR = Path("model_registry")


def register_policy(policy_path, name, metrics=None, source_config=None):
    src = Path(policy_path)
    if not src.is_file():
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    version = f"{name}-{int(time.time())}"
    dest_dir = REGISTRY_DIR / version
    safe_mkdir(str(dest_dir))
    dest_policy = dest_dir / src.name
    shutil.copy2(src, dest_policy)
    metadata = {
        "name": name,
        "version": version,
        "policy_path": str(dest_policy),
        "source_config": source_config,
        "metrics": metrics or {},
        "created_at": int(time.time()),
    }
    with open(dest_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    with open(REGISTRY_DIR / "latest.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata
