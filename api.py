"""FastAPI service for policy inference and monitoring."""

from pathlib import Path
from typing import Literal, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import QLearningAgent
from environment import ACTION_LABELS
from monitoring import evaluate_and_log_drift, log_prediction
from utils import discretise_battery


POLICY_PATH = Path("policy_v1.pkl")
TASK_TO_ID = {"low": 0, "medium": 1, "high": 2}

app = FastAPI(title="Smart Energy RL Policy API", version="1.0.0")
_agent = None


class PredictionRequest(BaseModel):
    battery_pct: float = Field(ge=0, le=100)
    task_demand: Literal["low", "medium", "high"]


class PredictionResponse(BaseModel):
    state: Tuple[int, int]
    action: int
    action_label: str


def get_agent():
    global _agent
    if _agent is None:
        if not POLICY_PATH.is_file():
            raise HTTPException(status_code=503, detail="Policy artifact is missing. Run train.py first.")
        agent = QLearningAgent()
        agent.load(str(POLICY_PATH))
        agent.epsilon = 0.0
        _agent = agent
    return _agent


@app.get("/health")
def health():
    return {"status": "ok", "policy_available": POLICY_PATH.is_file()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    agent = get_agent()
    state = (discretise_battery(payload.battery_pct), TASK_TO_ID[payload.task_demand])
    action = agent.choose_action(state)
    log_prediction(state, action, payload.battery_pct, payload.task_demand)
    return {"state": state, "action": action, "action_label": ACTION_LABELS[action]}


@app.get("/monitoring/drift")
def drift(limit: int = 500, threshold: float = 0.2):
    return evaluate_and_log_drift(limit=limit, threshold=threshold)
