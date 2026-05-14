# Local Model Registry

`train.py` writes versioned policy artifacts here after each run.

Generated files are ignored by Git:

- `model_registry/latest.json`
- `model_registry/<experiment>-<timestamp>/`

Keep this README so the registry location is visible in a fresh clone.
