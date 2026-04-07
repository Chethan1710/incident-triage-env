from fastapi import FastAPI, HTTPException
from env import IncidentEnv, Action, Observation, Reward
from scenarios import SCENARIOS
from grader import grade

app = FastAPI(title="AI Incident Triage Env")

env = IncidentEnv()
_current_task = None


@app.post("/reset")
def reset(task: str = "easy") -> Observation:
    global _current_task
    if task not in SCENARIOS:
        raise HTTPException(400, f"Unknown task: {task}. Choose from {list(SCENARIOS.keys())}")
    env.load_scenario(SCENARIOS[task])
    _current_task = task
    return env.reset()


@app.post("/step")
def step(action: Action):
    if env._state is None:
        raise HTTPException(400, "Call /reset first")
    obs, reward, done, info = env.step(action)
    score = None
    if done:
        score = grade(
            _current_task,
            info.get("correct", False),
            info.get("steps", env.steps),
            env.max_steps,
            obs.history,
        )
    return {"observation": obs, "reward": reward, "done": done, "info": info, "score": score}


@app.get("/state")
def state() -> Observation:
    if env._state is None:
        raise HTTPException(400, "Call /reset first")
    return env.state()


@app.get("/")
def root():
    return {"status": "ok", "tasks": list(SCENARIOS.keys())}