"""
Inference module for running local inference with the triage agent.
This module runs entirely offline without API calls.
"""

from environment import IncidentEnv
from tasks import SCENARIOS
from agent import TriageAgent
from graders import grade


def run_task(task_name: str, verbose: bool = True) -> dict:
    """
    Run a single task with the triage agent.

    Args:
        task_name: One of "easy", "medium", "hard"
        verbose: Whether to print progress

    Returns:
        dict with task results
    """
    if task_name not in SCENARIOS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(SCENARIOS.keys())}")

    env = IncidentEnv()
    env.load_scenario(SCENARIOS[task_name])
    agent = TriageAgent()

    obs = env.reset()
    agent.reset()
    done = False
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"[START] task={task_name} env=incident_triage model=triage_agent")

    while not done:
        step += 1
        action = agent.act(obs, step)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        if verbose:
            target = action.target if action.target is not None else "None"
            print(
                f"[STEP] step={step} action={action.action_type} "
                f"target={target} reward={reward.value:.2f} "
                f"done={str(done).lower()} error=null"
            )

    correct = info.get("correct", False)
    steps = info.get("steps", env.steps)
    score = grade(task_name, correct, steps, env.max_steps, obs.history)

    if verbose:
        print(
            f"[END] success={str(correct).lower()} "
            f"steps={steps} score={score:.4f} "
            f"rewards={total_reward:.2f}"
        )

    return {
        "task": task_name,
        "correct": correct,
        "steps": steps,
        "score": score,
        "total_reward": round(total_reward, 2),
    }


def run_all_tasks(verbose: bool = True) -> dict:
    """
    Run all tasks and return results.

    Returns:
        dict mapping task names to results
    """
    results = {}
    for task in ["easy", "medium", "hard"]:
        results[task] = run_task(task, verbose)
    return results


if __name__ == "__main__":
    run_all_tasks(verbose=True)
