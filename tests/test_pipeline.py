from agent import QLearningAgent
from api import app
from data_prep import clean_and_engineer, generate_raw_tasks
from fastapi.testclient import TestClient
from monitoring import compute_task_drift


def test_agent_policy_shape():
    agent = QLearningAgent()
    policy = agent.get_policy()
    assert len(policy) == 9
    assert set(policy.values()) == {0}


def test_data_cleaning_outputs_engineered_features(tmp_path):
    raw = tmp_path / "raw.csv"
    clean = tmp_path / "clean.csv"
    generate_raw_tasks(raw, rows=50, seed=7)
    _, n_rows = clean_and_engineer(raw, clean)
    assert n_rows > 0
    text = clean.read_text(encoding="utf-8")
    assert "battery_bucket" in text
    assert "is_high_demand" in text


def test_drift_detector_flags_shift():
    events = [{"task_demand": "high"} for _ in range(100)]
    metric = compute_task_drift(events, threshold=0.2)
    assert metric["drift_detected"] is True


def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
