"""
Custom AI Agent for Incident Triage
====================================
No external APIs. No ML libraries. Pure reasoning engine.

How it works:
  1. SIGNAL ANALYSIS  - scores every observable clue in the state
  2. HYPOTHESIS TABLE - tracks suspicion score per service
  3. ACTION PLANNER   - picks the highest-value action given current knowledge
  4. MEMORY           - avoids repeating actions, tracks what has been learned

Decision flow each step:
  -> If high-confidence hypothesis exists -> identify_root_cause
  -> Else if noisy alerts exist           -> filter_alerts
  -> Else if uninspected suspect exists   -> inspect_service
  -> Else if logs are messy              -> correlate_logs
  -> Else inspect next unknown service
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Signal weights
# Each alert type contributes to a service suspicion score.
ALERT_WEIGHTS = {
    "timeout":      10,
    "error_rate":    8,
    "db_error":     10,
    "connection":    9,
    "crash":        10,
    "deployment":    9,
    "config_error":  9,
    "high_latency":  4,   # downstream symptom, not root cause
    "cache_miss":    3,   # usually symptom
    "cpu_spike":     2,   # often noise
    "memory_spike":  2,   # often noise
}

# Log pattern -> (service hint, score boost)
LOG_PATTERNS = [
    (r"db|database|sql|postgres|mysql|mongo",        "database",       9),
    (r"connection.*(timeout|refused|reset)",          None,             7),
    (r"max connections|connection pool",              "database",       8),
    (r"slow query|lock timeout|deadlock",             "database",       9),
    (r"config.*(fail|error|unreachable|not found)",   "config_service", 9),
    (r"deployment.*(fail|error)",                     "config_service", 8),
    (r"version mismatch",                             "config_service", 7),
    (r"cache miss|eviction|cache fail",               "cache",          5),
    (r"retry|retrying",                               None,             3),
    (r"500|internal server error",                    None,             4),
    (r"memory.*(grow|leak|oom)",                      None,             2),
]

# Downstream services are symptoms, not root causes
DOWNSTREAM_PENALTY = {
    "frontend": -5,
    "api":      -3,
}

# How confident we need to be before declaring root cause
CONFIDENCE_THRESHOLD = 14


@dataclass
class AgentMemory:
    hypotheses: dict = field(default_factory=dict)  # service -> suspicion score
    inspected: set = field(default_factory=set)      # already inspected
    filtered: bool = False
    correlated: bool = False
    steps: int = 0


class TriageAgent:
    """
    Custom AI agent that reasons about system state to find root causes.
    Builds a suspicion score per service from alerts + logs + dependency graph,
    then acts on the highest-confidence hypothesis.
    """

    def __init__(self):
        self.memory = AgentMemory()

    def reset(self):
        self.memory = AgentMemory()

    def act(self, obs, step=0):
        from env import Action
        self.memory.steps += 1
        self._update_hypotheses(obs)
        return self._plan_action(obs)

    def _update_hypotheses(self, obs):
        scores = {}

        # Score from alerts
        for alert in obs.alerts:
            svc = alert.get("service")
            atype = alert.get("type", "")
            weight = ALERT_WEIGHTS.get(atype, 1)
            if svc:
                scores[svc] = scores.get(svc, 0) + weight

        # Score from log patterns
        for log in obs.logs:
            log_lower = log.lower()
            for pattern, hint_svc, boost in LOG_PATTERNS:
                if re.search(pattern, log_lower):
                    if hint_svc:
                        scores[hint_svc] = scores.get(hint_svc, 0) + boost
                    else:
                        for svc in obs.visible_services:
                            scores[svc] = scores.get(svc, 0) + (boost // 2)

        # Dependency propagation: if dep has high score, parent inherits some
        for svc, deps in obs.dependencies.items():
            for dep in deps:
                if dep in scores:
                    scores[svc] = scores.get(svc, 0) + scores[dep] * 0.3

        # Penalise known downstream/symptom services
        for svc, penalty in DOWNSTREAM_PENALTY.items():
            if svc in scores:
                scores[svc] += penalty

        # Boost inspected services (we have better info on them)
        for svc in self.memory.inspected:
            if svc in scores:
                scores[svc] += 2

        # Merge into persistent hypothesis table
        for svc, score in scores.items():
            prev = self.memory.hypotheses.get(svc, 0)
            self.memory.hypotheses[svc] = max(prev, score)

    def _plan_action(self, obs):
        from env import Action

        visible = set(obs.visible_services)
        all_services = set(obs.dependencies.keys())
        uninspected = all_services - visible - self.memory.inspected

        best_svc, best_score = self._top_hypothesis()

        # Step 1: Confident enough? Declare root cause.
        if best_score >= CONFIDENCE_THRESHOLD and best_svc in visible:
            return Action(action_type="identify_root_cause", target=best_svc)

        # Step 2: Filter noise if noisy alerts present
        alert_types = [a.get("type") for a in obs.alerts]
        has_noise = any(t in ("cpu_spike", "memory_spike") for t in alert_types)
        if has_noise and not self.memory.filtered:
            self.memory.filtered = True
            return Action(action_type="filter_alerts")

        # Step 3: Inspect the most suspicious uninspected service
        best_uninspected = self._top_uninspected(uninspected)
        if best_uninspected:
            self.memory.inspected.add(best_uninspected)
            return Action(action_type="inspect_service", target=best_uninspected)

        # Step 4: Correlate logs to reduce noise
        if not self.memory.correlated:
            self.memory.correlated = True
            return Action(action_type="correlate_logs")

        # Step 5: Commit to best hypothesis
        if best_svc:
            return Action(action_type="identify_root_cause", target=best_svc)

        # Fallback: inspect any remaining unknown service
        remaining = all_services - self.memory.inspected - visible
        if remaining:
            target = sorted(remaining)[0]
            self.memory.inspected.add(target)
            return Action(action_type="inspect_service", target=target)

        return Action(
            action_type="identify_root_cause",
            target=list(visible)[0] if visible else "unknown",
        )

    def _top_hypothesis(self):
        if not self.memory.hypotheses:
            return None, 0
        best = max(self.memory.hypotheses, key=self.memory.hypotheses.get)
        return best, self.memory.hypotheses[best]

    def _top_uninspected(self, uninspected):
        candidates = {svc: self.memory.hypotheses.get(svc, 0) for svc in uninspected}
        return max(candidates, key=candidates.get) if candidates else None

    def explain(self):
        """Human-readable reasoning state — useful for UI display."""
        ranked = sorted(self.memory.hypotheses.items(), key=lambda x: x[1], reverse=True)
        return {
            "top_suspect": ranked[0][0] if ranked else None,
            "confidence": round(ranked[0][1], 1) if ranked else 0,
            "all_suspects": {k: round(v, 1) for k, v in ranked},
            "inspected": list(self.memory.inspected),
            "steps": self.memory.steps,
        }