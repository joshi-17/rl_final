# GitOps And Collaboration Workflow

## Branches

- `main`: release-ready project state.
- `dev`: integration branch for completed features.
- `feature/<name>`: short-lived implementation branches.

## Pull Requests

Each PR should include:

- What changed.
- How it was tested.
- Any metric changes from `results.csv` or `outputs/index.html`.
- Screenshots or report links when plots changed.

## GitOps

Kubernetes desired state lives in `k8s/`. Changes to deployment configuration
should be reviewed through PRs. A GitOps controller such as Argo CD or Flux can
watch the repository and sync the manifests into the cluster.

## Rollback

Model rollback uses `model_registry/latest.json` or an earlier
`model_registry/<version>/metadata.json` entry. Platform rollback uses normal
Kubernetes rollout history:

```bash
kubectl rollout undo deployment/rl-energy-api
```
