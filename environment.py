from models import Observation, Action, Reward


class IncidentEnv:
    def __init__(self):
        self.scenario = None
        self._state = None
        self.steps = 0
        self.max_steps = 10

    def load_scenario(self, scenario: dict):
        self.scenario = scenario

    def reset(self) -> Observation:
        self.steps = 0
        self._state = {
            "alerts": list(self.scenario["alerts"]),
            "logs": list(self.scenario["logs"]),
            "visible_services": list(self.scenario["initial_visible"]),
            "dependencies": dict(self.scenario["dependencies"]),
            "history": [],
        }
        return Observation(**self._state)

    def state(self) -> Observation:
        return Observation(**self._state)

    def step(self, action: Action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        s = self._state
        act = action.action_type
        tgt = action.target

        if act == "inspect_service":
            if tgt and tgt not in s["visible_services"]:
                s["visible_services"].append(tgt)
                extra = self.scenario.get("service_logs", {}).get(tgt, [])
                s["logs"].extend(extra)
                reward = 2.0
            else:
                reward = -1.0

        elif act == "filter_alerts":
            noise = self.scenario.get("noise_alerts", [])
            before = len(s["alerts"])
            s["alerts"] = [a for a in s["alerts"] if a not in noise]
            reward = 1.0 if len(s["alerts"]) < before else -1.0

        elif act == "correlate_logs":
            s["logs"] = list(set(s["logs"]))
            reward = 1.0

        elif act == "identify_root_cause":
            correct = self.scenario["root_cause"]
            if tgt == correct:
                efficiency_bonus = max(0, (self.max_steps - self.steps) * 0.5)
                reward = 10.0 + efficiency_bonus
            else:
                reward = -5.0
            done = True
            info["correct"] = tgt == correct
            info["steps"] = self.steps

        else:
            reward = -1.0

        if self.steps >= self.max_steps and not done:
            done = True
            reward -= 3.0
            info["correct"] = False
            info["steps"] = self.steps

        s["history"].append({"action": act, "target": tgt, "reward": reward})
        return Observation(**s), Reward(value=reward), done, info
