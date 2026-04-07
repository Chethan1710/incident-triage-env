def compute_reward(action_type: str, target: str, scenario: dict, state: dict, steps: int, max_steps: int) -> float:
    """Standalone reward computation (used if you want to decouple from env.step)."""
    if action_type == "identify_root_cause":
        if target == scenario["root_cause"]:
            efficiency_bonus = max(0, (max_steps - steps) * 0.5)
            return 10.0 + efficiency_bonus
        return -5.0

    if action_type == "inspect_service":
        if target and target not in state["visible_services"]:
            return 2.0
        return -1.0

    if action_type == "filter_alerts":
        noise = scenario.get("noise_alerts", [])
        removable = [a for a in state["alerts"] if a in noise]
        return 1.0 if removable else -1.0

    if action_type == "correlate_logs":
        return 1.0

    return -1.0  # unknown action