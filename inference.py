"""
Inference module for running inference with the triage agent via LLM proxy.
"""

import os
import json
from openai import OpenAI

from environment import IncidentEnv
from tasks import SCENARIOS
from agent import TriageAgent
from graders import grade


# Safely read env vars
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "triage-agent")

if not API_BASE_URL or not API_KEY:
    print("[WARN] Missing API_BASE_URL or API_KEY — using local agent fallback")
    client = None
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def query_llm_for_action(obs, step, task_name):
    """
    Query the LLM proxy to decide the next action.
    Falls back to local agent if LLM call fails.
    """
    if client is None:
        return _local_fallback_action(obs, task_name)

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

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1]
            if content.startswith("json"):
                content = content[4:].strip()
        result = json.loads(content)
        return result.get("action_type", "inspect_service"), result.get("target")

    except Exception as e:
        print(f"[WARN] LLM call failed: {e} — using local fallback")
        return _local_fallback_action(obs, task_name)


def _local_fallback_action(obs, task_name):
    """
    Local fallback using rule-based agent when LLM is unavailable.
    """
    agent = TriageAgent()
    agent.reset()
    action = agent.act(obs, step=1)
    return action.action_type, action.target


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
    agent.reset()

    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    if verbose:
        print(f"[START] task={task_name} env=incident_triage model=triage_agent")

    while not done:
        step += 1
        action_type, target = query_llm_for_action(obs, step, task_name)
        from models import Action
        action = Action(action_type=action_type, target=target)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value

        if verbose:
            tgt = target if target is not None else "None"
            print(
                f"[STEP] step={step} action={action_type} "
                f"target={tgt} reward={reward.value:.2f} "
                f"done={str(done).lower()} error=null"
            )

        if step >= env.max_steps:
            done = True

    correct = info.get("correct", False)
    steps = info.get("steps", step)
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
        try:
            results[task] = run_task(task, verbose)
        except Exception as e:
            print(f"[ERROR] Task {task} failed: {e}")
            results[task] = {
                "task": task,
                "correct": False,
                "steps": 0,
                "score": 0.01,
                "total_reward": 0.0,
            }
    return results


if __name__ == "__main__":
    run_all_tasks(verbose=True)