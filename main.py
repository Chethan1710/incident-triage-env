<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, Request
=======
from fastapi import FastAPI, HTTPException
>>>>>>> e12b981e38929ad56abec1a80f58e6bac9cc38aa
from fastapi.middleware.cors import CORSMiddleware
from env import IncidentEnv, Action
from scenarios import SCENARIOS
from grader import grade

app = FastAPI(
    title="AI Incident Triage Env",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Fix Hugging Face proxy routing
@app.middleware("http")
async def root_path_middleware(request: Request, call_next):
    request.scope["root_path"] = ""
    response = await call_next(request)
    return response

# Enable CORS (important for validator + HF)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable CORS (important for HF + validator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = IncidentEnv()
_current_task = "easy"


@app.post("/reset")
def reset():
    global _current_task

<<<<<<< HEAD
    _current_task = "easy"  # validator does not send task
=======
    # Validator does NOT send task → default to easy
    _current_task = "easy"
>>>>>>> e12b981e38929ad56abec1a80f58e6bac9cc38aa
    env.load_scenario(SCENARIOS[_current_task])

    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: dict):
    if env._state is None:
        raise HTTPException(400, "Call /reset first")

    act = Action(**action)
    obs, reward, done, info = env.step(act)

    score = None
    if done:
        score = grade(
            _current_task,
            info.get("correct", False),
            info.get("steps", env.steps),
            env.max_steps,
            obs.history,
        )

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
        "score": score,
    }


@app.get("/state")
def state():
    if env._state is None:
        raise HTTPException(400, "Call /reset first")

    return env.state().dict()


@app.get("/")
def root():
<<<<<<< HEAD
    return {"message": "Incident Triage API is running"}
=======
    return {"status": "ok"}
>>>>>>> e12b981e38929ad56abec1a80f58e6bac9cc38aa
