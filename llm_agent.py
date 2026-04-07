import os
import json
import requests
from env import Action, Observation

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o-mini"  # more stable for structured output

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) triaging a production incident.

You must choose EXACTLY ONE action.

Actions:
- inspect_service(target)
- filter_alerts()
- correlate_logs()
- identify_root_cause(target)

Respond ONLY with valid JSON. No explanation.

Examples:
{"action_type": "inspect_service", "target": "database"}
{"action_type": "identify_root_cause", "target": "config_service"}
{"action_type": "filter_alerts", "target": null}
"""


def obs_to_prompt(obs: Observation, step: int) -> str:
    return f"""Step {step}

ALERTS:
{json.dumps(obs.alerts)}

LOGS:
{json.dumps(obs.logs)}

VISIBLE SERVICES:
{obs.visible_services}

DEPENDENCIES:
{json.dumps(obs.dependencies)}

HISTORY:
{json.dumps(obs.history)}

Return ONLY JSON action.
"""


class LLMAgent:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")

    def act(self, obs: Observation, step: int = 0) -> Action:
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_to_prompt(obs, step)},
                    ],
                    "temperature": 0,
                },
                timeout=30,
            )

            result = response.json()

            # 🔴 CRITICAL FIX: handle API errors properly
            if "choices" not in result:
                raise Exception(f"API Error: {result}")

            raw = result["choices"][0]["message"]["content"].strip()

            # Clean response
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            # Parse JSON safely
            data = json.loads(raw)

            return Action(
                action_type=data["action_type"],
                target=data.get("target"),
            )

        except Exception as e:
            raise Exception(f"LLM Agent failed: {str(e)}")