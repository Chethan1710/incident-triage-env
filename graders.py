def grade(task: str, correct: bool, steps: int, max_steps: int, decisions: list) -> float:
    correctness = 1.0 if correct else 0.0
    efficiency = max(0.0, 1.0 - (steps / max_steps))
    useful = sum(1 for d in decisions if d.get("reward", 0) > 0)
    total = len(decisions) if decisions else 1
    decision_quality = useful / total

    if task == "easy":
        raw = correctness
    elif task == "medium":
        raw = 0.7 * correctness + 0.3 * efficiency
    elif task == "hard":
        raw = 0.5 * correctness + 0.3 * efficiency + 0.2 * decision_quality
    else:
        raw = 0.6 * correctness + 0.4 * efficiency

    # Clamp to strictly between (0, 1)
    score = max(0.01, min(0.99, raw))
    return round(score, 4)