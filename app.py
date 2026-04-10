"""
Hugging Face Spaces Gradio UI for Incident Triage Environment
"""

import gradio as gr
import json

from environment import IncidentEnv
from tasks import SCENARIOS
from agent import TriageAgent
from graders import grade


def reset_env(scenario):
    """Reset the environment with selected scenario."""
    env = IncidentEnv()
    env.load_scenario(SCENARIOS[scenario])
    obs = env.reset()
    agent = TriageAgent()
    agent.reset()

    return format_observation(obs), "", scenario, "env_state", agent, env


def step_env(action_type, target, state_tuple):
    """Take a step in the environment."""
    env_state = state_tuple
    agent = None

    # Unpack state - last two are agent and env
    env = env_state[-1]
    agent = env_state[-2]
    scenario_key = env_state[-3]

    if env._state is None:
        return "Please reset the environment first!", "", state_tuple

    action_type = action_type.strip().lower().replace(" ", "_")
    target = target.strip() if target.strip() else None

    action = {"action_type": action_type, "target": target}

    try:
        from models import Action as ActionModel
        act = ActionModel(action_type=action_type, target=target)
        obs, reward, done, info = env.step(act)

        action_str = f"{action_type}"
        if target:
            action_str += f" -> {target}"

        reward_str = f"+{reward.value}" if reward.value >= 0 else str(reward.value)
        result = f"Action: {action_str}\nReward: {reward_str}\n"

        if done:
            final_score = grade(scenario_key, info.get("correct", False), info.get("steps", env.steps), env.max_steps, obs.history)
            result += f"\n[DONE] Correct: {info.get('correct', False)} | Score: {final_score:.3f}"

        return format_observation(obs), result, state_tuple
    except Exception as e:
        return f"Error: {str(e)}", "", state_tuple


def format_observation(obs):
    """Format observation for display."""
    lines = []
    lines.append("=" * 40)
    lines.append("ACTIVE ALERTS")
    lines.append("=" * 40)
    for alert in obs.alerts:
        lines.append(f"  [{alert.get('type', '?').upper()}] {alert.get('service', '?')}")

    lines.append("")
    lines.append("=" * 40)
    lines.append("VISIBLE SERVICES")
    lines.append("=" * 40)
    for svc in obs.visible_services:
        lines.append(f"  - {svc}")

    lines.append("")
    lines.append("=" * 40)
    lines.append("LOG ENTRIES")
    lines.append("=" * 40)
    for log in obs.logs[:10]:
        lines.append(f"  > {log[:80]}")
    if len(obs.logs) > 10:
        lines.append(f"  ... and {len(obs.logs) - 10} more")

    lines.append("")
    lines.append("=" * 40)
    lines.append("ACTION HISTORY")
    lines.append("=" * 40)
    for i, h in enumerate(obs.history):
        r = h.get('reward', 0)
        r_str = f"+{r}" if r >= 0 else str(r)
        tgt = h.get('target', '-')
        lines.append(f"  {i+1}. {h.get('action', '?')} ({tgt}) -> {r_str}")

    return "\n".join(lines)


def run_auto(scenario):
    """Run agent automatically to completion."""
    env = IncidentEnv()
    env.load_scenario(SCENARIOS[scenario])
    obs = env.reset()
    agent = TriageAgent()
    agent.reset()

    steps_log = []
    done = False
    step = 0

    while not done and step < env.max_steps:
        step += 1
        action = agent.act(obs, step)
        obs, reward, done, info = env.step(action)

        tgt = action.target if action.target else "-"
        r_str = f"+{reward.value}" if reward.value >= 0 else str(reward.value)
        steps_log.append(f"Step {step}: {action.action_type} ({tgt}) -> {r_str}")

    score = grade(scenario, info.get("correct", False), info.get("steps", env.steps), env.max_steps, obs.history)

    result = f"Scenario: {scenario.upper()}\n"
    result += f"Correct: {info.get('correct', False)}\n"
    result += f"Steps: {info.get('steps', step)}\n"
    result += f"Final Score: {score:.3f}\n"
    result += "\n" + "=" * 40 + "\n"
    result += "EXECUTION TRACE\n"
    result += "=" * 40 + "\n"
    result += "\n".join(steps_log)

    return format_observation(obs), result


# Gradio UI
with gr.Blocks(title="Incident Triage AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Incident Triage AI")
    gr.Markdown("Debug distributed system failures using AI-powered incident investigation.")

    with gr.Row():
        with gr.Column(scale=1):
            scenario_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Scenario"
            )
            reset_btn = gr.Button("Reset Environment", variant="primary")
            auto_run_btn = gr.Button("Run Agent (Auto)", variant="secondary")

        with gr.Column(scale=2):
            observation_display = gr.Textbox(
                label="Environment State",
                lines=20,
                interactive=False
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Take Action")
            action_type_input = gr.Dropdown(
                choices=["inspect_service", "filter_alerts", "correlate_logs", "identify_root_cause"],
                value="inspect_service",
                label="Action Type"
            )
            target_input = gr.Textbox(
                label="Target Service (optional)",
                placeholder="e.g., database, api, cache"
            )
            step_btn = gr.Button("Execute Action", variant="primary")

        with gr.Column(scale=1):
            result_display = gr.Textbox(
                label="Result",
                lines=10,
                interactive=False
            )

    # Store state across interactions
    env_state = gr.State()

    # Event handlers
    reset_btn.click(
        fn=reset_env,
        inputs=[scenario_dropdown],
        outputs=[observation_display, result_display, scenario_dropdown, env_state]
    )

    step_btn.click(
        fn=step_env,
        inputs=[action_type_input, target_input, env_state],
        outputs=[observation_display, result_display, env_state]
    )

    auto_run_btn.click(
        fn=run_auto,
        inputs=[scenario_dropdown],
        outputs=[observation_display, result_display]
    )

    gr.Markdown("""
    ---
    ### Action Types
    - **inspect_service** - Investigate a specific service for more clues
    - **filter_alerts** - Remove noise alerts to focus on real issues
    - **correlate_logs** - Find patterns across log entries
    - **identify_root_cause** - Declare the root cause of the incident

    ### Scoring
    - Correct root cause identification: 1.0
    - Efficiency bonus for fewer steps
    - Decision quality based on useful actions
    """)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
