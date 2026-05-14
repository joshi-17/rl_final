# Five-Minute Demo Script

1. Problem: explain adaptive energy management for IoT devices.
2. Data: run `python data_prep.py --rows 300` and show engineered features.
3. Training: run `python train.py --config configs/qlearning_v1.yaml --run_id demo`.
4. Tracking: show `results.csv`, `logs/demo.json`, `mlruns/`, and `model_registry/`.
5. Evaluation: run `python run_all.py --load_policy policy_v1.pkl --eval_episodes 200`.
6. Report: open `outputs/index.html` and discuss reward/battery behavior.
7. Serving: start `uvicorn api:app --host 127.0.0.1 --port 8000`.
8. Monitoring: call `/predict` several times and then `/monitoring/drift`.
9. Automation: show `.github/workflows/ci.yml`, `dvc.yaml`, and `k8s/`.
