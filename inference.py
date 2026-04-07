import os

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


from env import IncidentEnv
from scenarios import SCENARIOS
from agent import TriageAgent
from grader import grade


def run_task(task_name: str, verbose: bool = True) -> dict:
    env = IncidentEnv()
    env.load_scenario(SCENARIOS[task_name])
    agent = TriageAgent()

    obs = env.reset()
    agent.reset()
    done = False
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"[START] Task: {task_name}")

    while not done:
        step += 1
        action = agent.act(obs, step)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        if verbose:
            target = action.target if action.target is not None else "None"
            print(
                f"[STEP] action={action.action_type} "
                f"target={target} "
                f"reward={round(reward.value, 2)}"
            )

    correct = info.get("correct", False)
    steps = info.get("steps", env.steps)
    score = grade(task_name, correct, steps, env.max_steps, obs.history)

    if verbose:
        print(
            f"[END] correct={correct} "
            f"steps={steps} "
            f"score={round(score, 4)}"
        )

    return {
        "task": task_name,
        "correct": correct,
        "steps": steps,
        "score": score,
        "total_reward": round(total_reward, 2),
    }


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)