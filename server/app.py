from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import IncidentEnv
from models import Observation, Action, Reward
from tasks import SCENARIOS


app = FastAPI(title="Incident Triage Env", version="1.0")

# Global environment instance
_env: Optional[IncidentEnv] = None
_current_scenario: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ResetRequest(BaseModel):
    scenario: Optional[str] = "easy"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    global _env, _current_scenario

    scenario = request.scenario if request else "easy"

    if scenario not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario: {scenario}. Available: {list(SCENARIOS.keys())}"
        )

    _env = IncidentEnv()
    _env.load_scenario(SCENARIOS[scenario])
    _current_scenario = scenario

    obs = _env.reset()
    return JSONResponse(content={
        "alerts": obs.alerts,
        "logs": obs.logs,
        "visible_services": obs.visible_services,
        "dependencies": obs.dependencies,
        "history": obs.history,
    })


@app.post("/step")
async def step(request: StepRequest):
    global _env

    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    action_data = request.action
    action_type = action_data.get("action_type", "")
    target = action_data.get("target")

    if not action_type:
        raise HTTPException(
            status_code=400,
            detail="action_type is required"
        )

    action = Action(action_type=action_type, target=target)
    obs, reward, done, info = _env.step(action)

    return JSONResponse(content={
        "alerts": obs.alerts,
        "logs": obs.logs,
        "visible_services": obs.visible_services,
        "dependencies": obs.dependencies,
        "history": obs.history,
        "reward": reward.value,
        "done": done,
        "info": info,
    })


@app.get("/state")
async def state():
    global _env

    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    obs = _env.state()
    return JSONResponse(content={
        "alerts": obs.alerts,
        "logs": obs.logs,
        "visible_services": obs.visible_services,
        "dependencies": obs.dependencies,
        "history": obs.history,
    })


@app.get("/scenarios")
async def scenarios():
    return {
        "scenarios": list(SCENARIOS.keys()),
        "details": {
            key: {
                "label": val["label"],
                "tier": val["tier"],
                "description": val["description"],
            }
            for key, val in SCENARIOS.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
