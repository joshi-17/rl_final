# Architecture

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant DVC as DVC Pipeline
    participant Train as train.py
    participant MLflow as MLflow
    participant Registry as Local Registry
    participant API as FastAPI
    participant Monitor as Monitoring

    Dev->>DVC: dvc repro
    DVC->>DVC: prepare data
    DVC->>DVC: tune hyperparameters
    DVC->>Train: train selected config
    Train->>MLflow: log params, metrics, artifacts
    Train->>Registry: copy policy and metadata
    Registry->>API: load latest policy artifact
    API->>Monitor: write prediction logs
    Monitor->>Monitor: compute KL drift
```

The system is intentionally small but production-shaped: a deterministic
simulator provides reproducible training, the policy is tracked and registered,
the API serves predictions, and monitoring logs evidence for drift-triggered
retraining.
