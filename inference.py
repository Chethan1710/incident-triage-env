"""
Inference module for running inference with the triage agent via LLM proxy.
"""

import os
from openai import OpenAI

from environment import IncidentEnv
from tasks import SCENARIOS
from agent import TriageAgent
from graders import grade


# Initialize OpenAI client via proxy
client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"],
)


def query_llm_for_action(obs, step, task_name):
    """
    Query the LLM proxy to decide the next action.
    Makes a real API call through the proxy.
    """
    alerts_summary = ", ".join([f"{a['type']}@{a['service']}" for a in obs.alerts[:5]])
    visible = ", ".join(obs.visible_services)

    prompt = (
        f"You are an incident triage agent. Current step: {step}\n"
        f"Visible services: {visible}\n"
        f"Alerts: {alerts_summary}\n"
        f"Available actions: inspect_service, filter_alerts, correlate_logs, identify_root_cause\n"
        f"Decide the next action. Return ONLY JSON: {{\"action_type\": \"...\", \"target\": \"...\"}}\n"
        f"If uncertain, prefer: inspect_service with the most suspicious service."
    )

    response = client.chat.completions.create(
        model="triage-agent",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )

    import json
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:].strip()
    result = json.loads(content)
    return result.get("action_type", "inspect_service"), result.get("target")


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
        # Use LLM to decide action (makes API call through proxy)
        action_type, target = query_llm_for_action(obs, step, task_name)
        from models import Action
        action = Action(action_type=action_type, target=target)
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
