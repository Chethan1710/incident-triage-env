import streamlit as st
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    alerts: List[Dict]
    logs: List[str]
    visible_services: List[str]
    dependencies: Dict
    history: List[Dict]

@dataclass
class Action:
    action_type: str
    target: Optional[str] = None

@dataclass
class Reward:
    value: float

SCENARIOS = {
    "easy": {
        "alerts": [
            {"type": "timeout", "service": "database"},
            {"type": "high_latency", "service": "api"},
        ],
        "logs": ["DB connection timeout after 30s", "API retry attempt 1/3 failed", "Connection pool exhausted"],
        "initial_visible": ["api"],
        "dependencies": {"api": ["database"], "database": [], "cache": ["database"]},
        "service_logs": {"database": ["ERROR: max connections reached (limit: 100)", "Connection queue full"]},
        "noise_alerts": [],
        "root_cause": "database",
        "label": "Easy",
        "tier": "LOW",
        "description": "Single service failure. Signals are direct and unambiguous.",
    },
    "medium": {
        "alerts": [
            {"type": "high_latency", "service": "api"},
            {"type": "cache_miss", "service": "cache"},
            {"type": "timeout", "service": "database"},
            {"type": "cpu_spike", "service": "frontend"},
        ],
        "logs": ["Cache miss rate: 94%", "API response time: 2400ms", "DB slow query detected: 5.2s", "Frontend CPU: 87%"],
        "initial_visible": ["api", "cache"],
        "dependencies": {"frontend": ["api"], "api": ["database", "cache"], "database": [], "cache": ["database"]},
        "service_logs": {"database": ["Slow query log: SELECT * took 5.2s", "Lock timeout on table orders"]},
        "noise_alerts": [{"type": "cpu_spike", "service": "frontend"}],
        "root_cause": "database",
        "label": "Medium",
        "tier": "MED",
        "description": "Multiple alerts with noise. Root cause is one layer deep.",
    },
    "hard": {
        "alerts": [
            {"type": "high_latency", "service": "api"},
            {"type": "error_rate", "service": "frontend"},
            {"type": "memory_spike", "service": "api"},
        ],
        "logs": ["API heap growing: 2.1GB", "Frontend 500 errors: 43/min", "Config reload failed", "Stale config v1.2.0"],
        "initial_visible": ["frontend"],
        "dependencies": {"frontend": ["api"], "api": ["config_service", "database"], "config_service": [], "database": []},
        "service_logs": {
            "api": ["config_service unreachable: connection refused", "Falling back to stale config v1.2.0"],
            "config_service": ["FATAL: deployment rollout failed at 60%", "Version mismatch: expected 2.0, got 1.2"],
        },
        "noise_alerts": [{"type": "memory_spike", "service": "api"}],
        "root_cause": "config_service",
        "label": "Hard",
        "tier": "HIGH",
        "description": "Misleading signals. Root cause is hidden two hops away.",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class IncidentEnv:
    def __init__(self):
        self.scenario = None
        self._state = None
        self.steps = 0
        self.max_steps = 10

    def load_scenario(self, scenario):
        self.scenario = scenario

    def reset(self):
        self.steps = 0
        self._state = {
            "alerts": list(self.scenario["alerts"]),
            "logs": list(self.scenario["logs"]),
            "visible_services": list(self.scenario["initial_visible"]),
            "dependencies": dict(self.scenario["dependencies"]),
            "history": [],
        }
        return Observation(**self._state)

    def step(self, action):
        self.steps += 1
        reward = 0.0
        done = False
        info = {}
        s = self._state

        if action.action_type == "inspect_service":
            if action.target and action.target not in s["visible_services"]:
                s["visible_services"].append(action.target)
                extra = self.scenario.get("service_logs", {}).get(action.target, [])
                s["logs"].extend(extra)
                reward = 2.0
            else:
                reward = -1.0
        elif action.action_type == "filter_alerts":
            noise = self.scenario.get("noise_alerts", [])
            before = len(s["alerts"])
            s["alerts"] = [a for a in s["alerts"] if a not in noise]
            reward = 1.0 if len(s["alerts"]) < before else -1.0
        elif action.action_type == "correlate_logs":
            s["logs"] = list(set(s["logs"]))
            reward = 1.0
        elif action.action_type == "identify_root_cause":
            if action.target == self.scenario["root_cause"]:
                reward = 10.0 + max(0, (self.max_steps - self.steps) * 0.5)
            else:
                reward = -5.0
            done = True
            info["correct"] = action.target == self.scenario["root_cause"]
            info["steps"] = self.steps
        else:
            reward = -1.0

        if self.steps >= self.max_steps and not done:
            done = True
            reward -= 3.0
            info["correct"] = False
            info["steps"] = self.steps

        s["history"].append({"action": action.action_type, "target": action.target, "reward": reward})
        return Observation(**s), Reward(value=reward), done, info

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_WEIGHTS = {
    "timeout": 10, "error_rate": 8, "db_error": 10, "connection": 9,
    "crash": 10, "deployment": 9, "config_error": 9,
    "high_latency": 4, "cache_miss": 3, "cpu_spike": 2, "memory_spike": 2,
}
LOG_PATTERNS = [
    (r"db|database|sql|postgres|mysql",              "database",       9),
    (r"connection.*(timeout|refused|reset)",          "database",       7),
    (r"max connections|connection pool|exhausted",    "database",       8),
    (r"slow query|lock timeout|deadlock",             "database",       9),
    (r"config.*(fail|error|unreachable|mismatch)",    "config_service", 9),
    (r"deployment.*(fail|rollout)",                   "config_service", 8),
    (r"version mismatch|stale config",               "config_service", 7),
    (r"cache miss|eviction",                         "cache",          5),
    (r"retry|retrying",                              None,             3),
    (r"500|internal server error",                   None,             4),
    (r"memory.*(grow|leak|heap)",                    None,             2),
]
DOWNSTREAM_PENALTY = {"frontend": -5, "api": -3}
CONFIDENCE_THRESHOLD = 14

class TriageAgent:
    def __init__(self): self.reset()

    def reset(self):
        self.hypotheses: Dict[str, float] = {}
        self.inspected: set = set()
        self.filtered = False
        self.correlated = False
        self.steps = 0

    def act(self, obs, step=0):
        self.steps += 1
        self._update_hypotheses(obs)
        return self._plan(obs)

    def _update_hypotheses(self, obs):
        scores: Dict[str, float] = {}
        for a in obs.alerts:
            svc, atype = a.get("service"), a.get("type", "")
            if svc:
                scores[svc] = scores.get(svc, 0) + ALERT_WEIGHTS.get(atype, 1)
        for log in obs.logs:
            ll = log.lower()
            for pat, hint, boost in LOG_PATTERNS:
                if re.search(pat, ll):
                    if hint:
                        scores[hint] = scores.get(hint, 0) + boost
                    else:
                        for svc in obs.visible_services:
                            scores[svc] = scores.get(svc, 0) + boost // 2
        for svc, deps in obs.dependencies.items():
            for dep in deps:
                if dep in scores:
                    scores[svc] = scores.get(svc, 0) + scores[dep] * 0.3
        for svc, pen in DOWNSTREAM_PENALTY.items():
            if svc in scores:
                scores[svc] += pen
        for svc in self.inspected:
            if svc in scores:
                scores[svc] += 2
        for svc, s in scores.items():
            self.hypotheses[svc] = max(self.hypotheses.get(svc, 0), s)

    def _plan(self, obs):
        visible = set(obs.visible_services)
        all_svcs = set(obs.dependencies.keys())
        uninspected = all_svcs - visible - self.inspected
        best, score = self._top()

        if score >= CONFIDENCE_THRESHOLD and best in visible:
            return Action("identify_root_cause", best)
        atypes = [a.get("type") for a in obs.alerts]
        if any(t in ("cpu_spike", "memory_spike") for t in atypes) and not self.filtered:
            self.filtered = True
            return Action("filter_alerts")
        top_u = self._top_uninspected(uninspected)
        if top_u:
            self.inspected.add(top_u)
            return Action("inspect_service", top_u)
        if not self.correlated:
            self.correlated = True
            return Action("correlate_logs")
        if best:
            return Action("identify_root_cause", best)
        rem = all_svcs - self.inspected - visible
        if rem:
            t = sorted(rem)[0]; self.inspected.add(t)
            return Action("inspect_service", t)
        return Action("identify_root_cause", list(visible)[0] if visible else "unknown")

    def _top(self):
        if not self.hypotheses: return None, 0
        b = max(self.hypotheses, key=self.hypotheses.get)
        return b, self.hypotheses[b]

    def _top_uninspected(self, u):
        c = {s: self.hypotheses.get(s, 0) for s in u}
        return max(c, key=c.get) if c else None

    def explain(self):
        ranked = sorted(self.hypotheses.items(), key=lambda x: x[1], reverse=True)
        return {
            "top_suspect": ranked[0][0] if ranked else None,
            "confidence": round(ranked[0][1], 1) if ranked else 0,
            "all_suspects": {k: round(v, 1) for k, v in ranked},
        }

def grade(task, correct, steps, max_steps, decisions):
    c = 1.0 if correct else 0.0
    e = max(0.0, 1.0 - (steps / max_steps))
    useful = sum(1 for d in decisions if d.get("reward", 0) > 0)
    dq = useful / len(decisions) if decisions else 0
    if task == "easy":   return round(c, 3)
    if task == "medium": return round(0.7 * c + 0.3 * e, 3)
    if task == "hard":   return round(0.5 * c + 0.3 * e + 0.2 * dq, 3)
    return round(0.6 * c + 0.4 * e, 3)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:     #070b14; --panel:  #0d1320; --card:   #111827;
    --border: #1e2d45; --bhi:    #2a3f5f;
    --cyan:   #00d4ff; --cyand:  #00d4ff22; --cyag:   #00d4ff44;
    --green:  #00ff88; --greend: #00ff8822;
    --red:    #ff3d71; --redd:   #ff3d7122;
    --amber:  #ffaa00; --amberd: #ffaa0022;
    --purple: #a78bfa;
    --hi: #f0f4ff; --mid: #8899bb; --lo: #3d5070;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Syne', sans-serif;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--hi)!important;font-family:var(--sans)!important;}
.stApp{background:var(--bg)!important;}.stApp>header{background:transparent!important;}
#MainMenu,footer,.stDeployButton{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--panel);}
::-webkit-scrollbar-thumb{background:var(--bhi);border-radius:2px;}
section[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{font-family:var(--sans)!important;}
.stButton>button{background:var(--cyan)!important;color:#000!important;border:none!important;border-radius:6px!important;font-family:var(--mono)!important;font-weight:700!important;font-size:12px!important;letter-spacing:1px!important;transition:all .2s!important;box-shadow:0 0 20px var(--cyag)!important;}
.stButton>button:hover{background:#33deff!important;box-shadow:0 0 32px var(--cyan)!important;transform:translateY(-1px)!important;}
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--hi)!important;border-radius:6px!important;font-family:var(--mono)!important;font-size:12px!important;}
.stCheckbox label{color:var(--mid)!important;font-family:var(--mono)!important;font-size:12px!important;}
.stProgress>div>div>div{background:var(--cyan)!important;}.stProgress>div>div{background:var(--border)!important;}

@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.5;}}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
@keyframes countIn{from{opacity:0;transform:scale(.85);}to{opacity:1;transform:scale(1);}}

.topbar{background:var(--panel);border-bottom:1px solid var(--border);padding:14px 32px;display:flex;align-items:center;gap:20px;font-family:var(--mono);}
.topbar-logo{font-family:var(--sans);font-size:15px;font-weight:800;color:var(--cyan);display:flex;align-items:center;gap:8px;}
.topbar-logo::before{content:'';display:inline-block;width:7px;height:7px;background:var(--cyan);border-radius:50%;animation:pulse 2s infinite;}
.topbar-sep{color:var(--bhi);}
.topbar-tag{font-size:10px;color:var(--lo);letter-spacing:2px;text-transform:uppercase;}
.topbar-right{margin-left:auto;font-size:10px;color:var(--mid);}
.topbar-dot{display:inline-block;width:5px;height:5px;background:var(--green);border-radius:50%;margin-right:5px;animation:pulse 1.5s infinite;}

.hero{background:linear-gradient(135deg,#0d1320 0%,#070b14 60%,#0a0f1e 100%);border-bottom:1px solid var(--border);padding:32px;position:relative;overflow:hidden;animation:fadeUp .4s ease;}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);}
.hero-grid{display:grid;grid-template-columns:1fr 1fr 1fr 220px;gap:24px;align-items:center;}
.hero-label{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:3px;text-transform:uppercase;color:var(--lo);margin-bottom:6px;}
.hero-score{font-family:var(--mono);font-size:60px;font-weight:700;line-height:1;animation:countIn .6s cubic-bezier(.34,1.56,.64,1);}
.hero-score.ok{color:var(--green);text-shadow:0 0 40px var(--green);}
.hero-score.fail{color:var(--red);text-shadow:0 0 40px var(--red);}
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:4px;font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-top:10px;}
.badge.ok{background:var(--greend);border:1px solid var(--green);color:var(--green);}
.badge.fail{background:var(--redd);border:1px solid var(--red);color:var(--red);}
.hero-val{font-family:var(--mono);font-size:26px;font-weight:700;color:var(--hi);}
.hero-sub{font-family:var(--mono);font-size:11px;color:var(--mid);margin-top:4px;}

.rc-card{background:linear-gradient(135deg,#1a0a0a,#120810);border:1px solid var(--red);border-radius:10px;padding:20px 22px;position:relative;overflow:hidden;animation:fadeUp .7s ease;}
.rc-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--red),transparent);}
.rc-label{font-family:var(--mono);font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--red);margin-bottom:8px;display:flex;align-items:center;gap:6px;}
.rc-label::before{content:'⬤';font-size:7px;animation:pulse 1.5s infinite;}
.rc-name{font-family:var(--mono);font-size:20px;font-weight:700;color:#fff;}
.rc-sub{font-family:var(--mono);font-size:10px;color:var(--mid);margin-top:4px;}

.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:14px;}
.ph{background:var(--card);border-bottom:1px solid var(--border);padding:10px 18px;display:flex;align-items:center;justify-content:space-between;}
.pt{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--mid);display:flex;align-items:center;gap:8px;}
.pt-bar{display:inline-block;width:3px;height:13px;background:var(--cyan);border-radius:2px;}
.pb{padding:14px 18px;}
.ph-tag{font-family:var(--mono);font-size:10px;color:var(--lo);}

.step{display:flex;align-items:flex-start;gap:12px;padding:10px 14px;border-radius:7px;margin-bottom:7px;border:1px solid var(--border);background:var(--card);animation:fadeUp .3s ease;}
.step.s-ok{background:linear-gradient(135deg,#001a0a,#0d1320);border-color:var(--green);}
.step.s-fail{background:linear-gradient(135deg,#1a000a,#0d1320);border-color:var(--red);}
.sn{width:26px;height:26px;border-radius:5px;background:var(--bg);border:1px solid var(--bhi);display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:10px;font-weight:700;color:var(--mid);flex-shrink:0;}
.sc{flex:1;min-width:0;}
.sa{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--hi);}
.sa .an{color:var(--cyan);}
.sa .at{color:var(--amber);}
.sm{font-family:var(--mono);font-size:10px;color:var(--mid);margin-top:3px;}
.sr{font-family:var(--mono);font-size:11px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0;}
.sr.pos{background:var(--greend);color:var(--green);border:1px solid var(--green);}
.sr.neg{background:var(--redd);color:var(--red);border:1px solid var(--red);}

.hr{margin-bottom:12px;}
.hh{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}
.hn{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--hi);display:flex;align-items:center;gap:7px;}
.htag{font-size:8px;padding:2px 5px;border-radius:3px;font-weight:700;letter-spacing:1px;text-transform:uppercase;}
.htag.root{background:var(--greend);color:var(--green);border:1px solid var(--green);}
.htag.wrong{background:var(--redd);color:var(--red);border:1px solid var(--red);}
.htag.sym{background:var(--amberd);color:var(--amber);border:1px solid var(--amber);}
.hs{font-family:var(--mono);font-size:11px;color:var(--mid);}
.bt{background:var(--bg);border-radius:3px;height:5px;border:1px solid var(--border);}
.bf{border-radius:3px;height:5px;position:relative;}
.bf::after{content:'';position:absolute;right:0;top:-2px;width:2px;height:9px;background:inherit;filter:brightness(2);border-radius:2px;}

.tr{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid var(--border);align-items:flex-start;}
.tr:last-child{border-bottom:none;}
.ts{font-family:var(--mono);font-size:9px;color:var(--lo);width:44px;flex-shrink:0;padding-top:1px;}
.tsusp{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--cyan);}
.tconf{font-family:var(--mono);font-size:10px;color:var(--mid);}
.tact{font-family:var(--mono);font-size:10px;color:var(--hi);}

.gn{padding:7px 14px;border-radius:6px;font-family:var(--mono);font-size:11px;font-weight:600;border:1px solid var(--border);background:var(--card);color:var(--mid);white-space:nowrap;}
.gn.hl{border-color:var(--red);background:var(--redd);color:var(--red);box-shadow:0 0 14px var(--redd);}
.gn.sus{border-color:var(--amber);background:var(--amberd);color:var(--amber);}
.gn.clr{opacity:.4;}
.ga{color:var(--lo);font-family:var(--mono);font-size:13px;padding:0 6px;}

.al{display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:7px;border:1px solid var(--border);background:var(--card);margin-bottom:7px;}
.al.crit{border-left:3px solid var(--red);}
.al.warn{border-left:3px solid var(--amber);}
.al.info{border-left:3px solid var(--cyan);}
.al.noise{border-left:3px solid var(--lo);opacity:.45;}
.at2{font-family:var(--mono);font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;flex:1;}
.asvc{font-family:var(--mono);font-size:10px;color:var(--mid);padding:2px 7px;background:var(--bg);border:1px solid var(--border);border-radius:4px;}

.log-wrap{max-height:220px;overflow-y:auto;padding:2px 0;}
.ll{padding:6px 10px;border-radius:4px;font-family:var(--mono);font-size:10px;color:var(--mid);margin-bottom:3px;background:var(--bg);border:1px solid var(--border);display:flex;align-items:flex-start;gap:8px;}
.ll:hover{border-color:var(--bhi);color:var(--hi);}
.lpfx{color:var(--lo);flex-shrink:0;}
.ltag{font-size:8px;padding:1px 4px;border-radius:2px;background:var(--cyand);color:var(--cyan);flex-shrink:0;letter-spacing:1px;text-transform:uppercase;}

.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 10px;border-radius:5px;font-family:var(--mono);font-size:11px;font-weight:600;border:1px solid var(--border);background:var(--card);color:var(--mid);margin:3px;}
.chip.vis{border-color:var(--cyan);background:var(--cyand);color:var(--cyan);}
.chip.root{border-color:var(--red);background:var(--redd);color:var(--red);}
.chip-dot{width:4px;height:4px;border-radius:50%;background:currentColor;}

.sbar{font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--lo);margin:18px 0 9px;padding-bottom:5px;border-bottom:1px solid var(--border);}
.tier{display:inline-block;padding:3px 7px;border-radius:3px;font-family:var(--mono);font-size:9px;font-weight:700;letter-spacing:1px;}
.tier.LOW{background:var(--greend);color:var(--green);border:1px solid var(--green);}
.tier.MED{background:var(--amberd);color:var(--amber);border:1px solid var(--amber);}
.tier.HIGH{background:var(--redd);color:var(--red);border:1px solid var(--red);}

.empty{text-align:center;padding:60px 32px;animation:fadeUp .5s ease;}
.eicon{font-size:44px;opacity:.2;margin-bottom:14px;}
.etitle{font-family:var(--sans);font-size:20px;font-weight:700;color:var(--hi);margin-bottom:8px;}
.esub{font-family:var(--mono);font-size:12px;color:var(--mid);line-height:1.8;max-width:480px;margin:0 auto;}
.scard{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px 22px;margin-bottom:10px;}
.scard-name{font-family:var(--sans);font-size:16px;font-weight:700;color:var(--hi);margin:7px 0 5px;}
.scard-desc{font-family:var(--mono);font-size:11px;color:var(--mid);line-height:1.6;}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    "timeout":      ("crit", "TIMEOUT",      "◈"),
    "error_rate":   ("crit", "ERROR RATE",   "◈"),
    "high_latency": ("warn", "HIGH LATENCY", "◇"),
    "cache_miss":   ("warn", "CACHE MISS",   "◇"),
    "cpu_spike":    ("noise","CPU SPIKE",    "·"),
    "memory_spike": ("noise","MEM SPIKE",    "·"),
    "deployment":   ("crit", "DEPLOYMENT",   "◈"),
}

ACTION_COLOR = {
    "inspect_service":    "var(--cyan)",
    "filter_alerts":      "var(--amber)",
    "correlate_logs":     "var(--purple)",
    "identify_root_cause":"var(--green)",
}
ACTION_LABEL = {
    "inspect_service":    "INSPECT",
    "filter_alerts":      "FILTER",
    "correlate_logs":     "CORRELATE",
    "identify_root_cause":"IDENTIFY",
}

def tag_log(log):
    ll = log.lower()
    if any(k in ll for k in ["db","database","sql","query","connection","pool"]): return "database"
    if any(k in ll for k in ["config","deployment","version","stale"]): return "config_service"
    if "cache" in ll: return "cache"
    if any(k in ll for k in ["api","request","response"]): return "api"
    return None

def causal_path(root, deps):
    path = []
    def dfs(node, visited):
        path.append(node)
        visited.add(node)
        for svc, d in deps.items():
            if node in d and svc not in visited:
                dfs(svc, visited)
    dfs(root, set())
    return path

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_topbar():
    st.markdown("""<div class="topbar">
    <div class="topbar-logo">TRIAGE.AI</div>
    <span class="topbar-sep">|</span>
    <span class="topbar-tag">Incident Intelligence Platform</span>
    <div class="topbar-right"><span class="topbar-dot"></span>SYSTEM NOMINAL</div>
    </div>""", unsafe_allow_html=True)


def render_hero(res, task_key):
    sc = SCENARIOS[task_key]
    ok = res["correct"]
    cls = "ok" if ok else "fail"
    eff = round((1 - res["steps"] / 10) * 100)
    verdict = "RESOLVED" if ok else "FAILED"
    root = res["root_cause"]
    guess = res["agent_answer"]
    meta = f"Agent identified correctly" if ok else f"Agent guessed: {guess}"

    st.markdown(f"""<div class="hero">
    <div class="hero-grid">
        <div>
            <div class="hero-label">Session Score</div>
            <div class="hero-score {cls}">{res['score']}</div>
            <div class="badge {cls}">{verdict}</div>
        </div>
        <div>
            <div class="hero-label">Steps Taken</div>
            <div class="hero-val">{res['steps']}<span style="font-size:14px;color:var(--lo)"> / 10</span></div>
            <div class="hero-sub">Efficiency: {eff}%</div>
        </div>
        <div>
            <div class="hero-label">Scenario</div>
            <div class="hero-val" style="font-size:19px">{sc['label']}</div>
            <div class="hero-sub" style="margin-top:6px"><span class="tier {sc['tier']}">{sc['tier']} SEVERITY</span></div>
        </div>
        <div class="rc-card">
            <div class="rc-label">Root Cause</div>
            <div class="rc-name">{root.upper()}</div>
            <div class="rc-sub">{meta}</div>
        </div>
    </div>
    </div>""", unsafe_allow_html=True)


def render_alerts(alerts, noise_alerts):
    noise_set = {(a["type"], a["service"]) for a in noise_alerts}
    html = ""
    for a in alerts:
        atype, svc = a.get("type",""), a.get("service","")
        noise = (atype, svc) in noise_set
        cfg = ALERT_CONFIG.get(atype, ("info","ALERT","·"))
        css = "noise" if noise else {"crit":"crit","warn":"warn","info":"info"}.get(cfg[0],"info")
        noise_note = ' <span style="font-size:9px;color:var(--lo)">[NOISE]</span>' if noise else ""
        html += f'<div class="al {css}"><span style="font-size:13px">{cfg[2]}</span><span class="at2" style="{"color:var(--lo)" if noise else ""}">{cfg[1]}{noise_note}</span><span class="asvc">{svc}</span></div>'
    st.markdown(f"""<div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>ACTIVE ALERTS</div><span class="ph-tag">{len(alerts)} FIRING</span></div><div class="pb">{html}</div></div>""", unsafe_allow_html=True)


def render_services(visible, dependencies, root_cause):
    chips = ""
    for svc in dependencies:
        cls = "root" if svc == root_cause else ("vis" if svc in visible else "")
        chips += f'<span class="chip {cls}"><span class="chip-dot"></span>{svc}</span>'
    deps_html = "".join(
        f'<div class="ll"><span class="lpfx">DEP</span>{s} → {", ".join(d)}</div>'
        for s, d in dependencies.items() if d
    )
    st.markdown(f"""<div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>SERVICE MAP</div></div><div class="pb"><div style="margin-bottom:12px">{chips}</div>{deps_html}</div></div>""", unsafe_allow_html=True)


def render_logs(logs, dependencies):
    lines = ""
    for log in logs:
        tag = tag_log(log)
        tag_html = f'<span class="ltag">{tag}</span>' if tag else ""
        lines += f'<div class="ll"><span class="lpfx">&gt;_</span>{tag_html}{log}</div>'
    st.markdown(f"""<div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>LOG STREAM</div><span class="ph-tag">{len(logs)} ENTRIES</span></div><div class="pb"><div class="log-wrap">{lines}</div></div></div>""", unsafe_allow_html=True)


def render_action_trace(steps_log):
    items = ""
    for s in steps_log:
        act, tgt, rew = s["action"], s["target"], s["reward"]
        is_final = act == "identify_root_cause"
        fcls = ("s-ok" if rew > 0 else "s-fail") if is_final else ""
        col = ACTION_COLOR.get(act, "var(--cyan)")
        lbl = ACTION_LABEL.get(act, act.upper())
        thtml = f' <span class="at">({tgt})</span>' if tgt and tgt != "—" else ""
        rcls = "pos" if rew >= 0 else "neg"
        rstr = f"+{rew}" if rew >= 0 else str(rew)
        items += f"""<div class="step {fcls}"><div class="sn">{s['step']:02d}</div><div class="sc"><div class="sa"><span class="an" style="color:{col}">{lbl}</span>{thtml}</div><div class="sm">{act.replace('_',' ')}</div></div><span class="sr {rcls}">{rstr}</span></div>"""
    st.markdown(f"""<div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>EXECUTION TRACE</div><span class="ph-tag">{len(steps_log)} STEPS</span></div><div class="pb">{items}</div></div>""", unsafe_allow_html=True)


def render_reasoning(reasoning_log, steps_log, root_cause, agent_answer):
    if not reasoning_log: return
    snap = reasoning_log[-1]
    suspects = snap.get("all_suspects", {})
    mx = max(suspects.values()) if suspects else 1

    bars = ""
    for svc, score in suspects.items():
        pct = int((score / mx) * 100) if mx else 0
        is_root = svc == root_cause
        is_wrong = svc == agent_answer and not is_root
        is_sym = svc in ("frontend", "api")
        col = "var(--green)" if is_root else ("var(--red)" if is_wrong else ("var(--lo)" if is_sym else "var(--cyan)"))
        tag = ('<span class="htag root">ROOT CAUSE</span>' if is_root else
               '<span class="htag wrong">WRONG GUESS</span>' if is_wrong else
               '<span class="htag sym">SYMPTOM</span>' if is_sym else "")
        bars += f'<div class="hr"><div class="hh"><div class="hn">{svc} {tag}</div><div class="hs">{score}</div></div><div class="bt"><div class="bf" style="width:{pct}%;background:{col}"></div></div></div>'

    traces = ""
    for i, (snap2, step) in enumerate(zip(reasoning_log, steps_log)):
        top = snap2.get("top_suspect","—")
        conf = snap2.get("confidence", 0)
        act = step["action"].replace("_"," ")
        tgt = step["target"] if step["target"] != "—" else ""
        traces += f'<div class="tr"><div class="ts">STEP {i+1:02d}</div><div><div><span class="tsusp">{top}</span> <span class="tconf">score:{conf}</span></div><div style="margin-top:2px"><span style="color:var(--lo)">→ </span><span class="tact">{act}{f" ({tgt})" if tgt else ""}</span></div></div></div>'

    st.markdown(f"""
    <div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>HYPOTHESIS TABLE</div></div><div class="pb">{bars}</div></div>
    <div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>REASONING TRACE</div></div><div class="pb">{traces}</div></div>
    """, unsafe_allow_html=True)


def render_causal_graph(dependencies, root_cause, show_chain):
    path = causal_path(root_cause, dependencies) if show_chain else []
    rows = ""
    seen = set()
    for svc, deps in dependencies.items():
        for dep in deps:
            if (svc, dep) in seen: continue
            seen.add((svc, dep))
            s_cls = "hl" if dep == root_cause else ("sus" if svc in path and show_chain else "")
            d_cls = "hl" if dep == root_cause else ("sus" if dep in path and show_chain else "clr" if show_chain else "")
            rows += f'<div style="display:flex;align-items:center;margin-bottom:8px"><div class="gn {s_cls}">{svc}</div><div class="ga">→</div><div class="gn {d_cls}">{dep}</div></div>'
    note = f'<div style="font-family:var(--mono);font-size:10px;color:var(--mid);margin-bottom:10px">Causal path highlighted → root: <span style="color:var(--red)">{root_cause}</span></div>' if show_chain and path else ""
    st.markdown(f"""<div class="panel"><div class="ph"><div class="pt"><div class="pt-bar"></div>CAUSAL GRAPH</div></div><div class="pb">{note}{rows}</div></div>""", unsafe_allow_html=True)


def render_empty():
    st.markdown("""<div class="empty"><div class="eicon">⬡</div>
    <div class="etitle">No Active Investigation</div>
    <div class="esub">Select a scenario from the sidebar and click<br><span style="color:var(--cyan)">RUN INVESTIGATION</span> to watch the AI agent triage the incident.</div></div>""", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (k, sc) in zip(cols, SCENARIOS.items()):
        with col:
            st.markdown(f'<div class="scard"><span class="tier {sc["tier"]}">{sc["tier"]}</span><div class="scard-name">{sc["label"]}</div><div class="scard-desc">{sc["description"]}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="TRIAGE.AI", page_icon="⬡", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    for k in ["results","steps_log","final_obs","reasoning_log","initial_alerts","initial_logs","task_key"]:
        if k not in st.session_state:
            st.session_state[k] = (None if k in ["results","final_obs"] else
                                   [] if k in ["steps_log","reasoning_log","initial_alerts","initial_logs"] else "easy")

    # SIDEBAR
    with st.sidebar:
        st.markdown('<div style="padding:18px 0 0"><span style="font-family:var(--mono);font-size:16px;font-weight:700;color:var(--cyan)">TRIAGE.AI</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sbar">Investigation Config</div>', unsafe_allow_html=True)
        task_key = st.selectbox("Scenario", list(SCENARIOS.keys()),
            format_func=lambda k: f"{SCENARIOS[k]['label']} — {SCENARIOS[k]['tier']}",
            index=list(SCENARIOS.keys()).index(st.session_state.task_key))
        st.session_state.task_key = task_key
        sc = SCENARIOS[task_key]
        st.markdown(f'<div style="font-family:var(--mono);font-size:11px;color:var(--mid);line-height:1.7;padding:6px 0">{sc["description"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sbar">Display Options</div>', unsafe_allow_html=True)
        show_chain = st.checkbox("Highlight causal chain", value=True)
        show_trace = st.checkbox("Show reasoning details", value=True)

        st.markdown('<div class="sbar">Actions</div>', unsafe_allow_html=True)
        run_btn = st.button("RUN INVESTIGATION", use_container_width=True)
        if st.button("RESET", use_container_width=True):
            for k in ["results","steps_log","final_obs","reasoning_log","initial_alerts","initial_logs"]:
                st.session_state[k] = None if k in ["results","final_obs"] else []
            st.rerun()

        st.markdown('<div class="sbar">Reward Reference</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:var(--mono);font-size:10px;color:var(--mid);line-height:2.2">
<span style="color:var(--green)">+10</span>  correct root cause<br>
<span style="color:var(--green)">+0.5×</span> remaining steps<br>
<span style="color:var(--cyan)">+2</span>  useful inspection<br>
<span style="color:var(--amber)">+1</span>  noise filtered<br>
<span style="color:var(--red)">-5</span>   wrong diagnosis<br>
<span style="color:var(--red)">-3</span>   timeout penalty</div>""", unsafe_allow_html=True)

    # RUN
    if run_btn:
        scenario = SCENARIOS[task_key]
        env = IncidentEnv()
        env.load_scenario(scenario)
        agent = TriageAgent()
        obs = env.reset()
        agent.reset()
        st.session_state.initial_alerts = list(obs.alerts)
        st.session_state.initial_logs = list(obs.logs)
        done = False
        steps_log, reasoning_log, step = [], [], 0
        prog_ph = st.empty()
        with prog_ph:
            prog = st.progress(0, text="Initializing investigation...")
        while not done and step < env.max_steps:
            step += 1
            time.sleep(0.12)
            action = agent.act(obs, step)
            snap = agent.explain()
            obs, reward, done, info = env.step(action)
            steps_log.append({"step":step,"action":action.action_type,"target":action.target or "—","reward":reward.value})
            reasoning_log.append(snap)
            prog.progress(min(step/env.max_steps, .95), text=f"Step {step:02d} — {action.action_type.replace('_',' ')}...")
        prog.progress(1.0, text="Investigation complete.")
        time.sleep(0.25)
        prog_ph.empty()
        correct = info.get("correct", False)
        steps_taken = info.get("steps", env.steps)
        score = grade(task_key, correct, steps_taken, env.max_steps, obs.history)
        st.session_state.results = {
            "correct": correct, "steps": steps_taken, "score": score, "task": task_key,
            "root_cause": scenario["root_cause"],
            "agent_answer": steps_log[-1]["target"] if steps_log else "—",
        }
        st.session_state.steps_log = steps_log
        st.session_state.final_obs = obs
        st.session_state.reasoning_log = reasoning_log
        st.rerun()

    # RENDER
    render_topbar()

    if st.session_state.results and st.session_state.final_obs:
        res = st.session_state.results
        obs = st.session_state.final_obs
        scenario = SCENARIOS[res["task"]]

        render_hero(res, res["task"])

        st.markdown('<div style="padding:20px 28px 0">', unsafe_allow_html=True)
        l, m, r = st.columns([1, 1.05, 1], gap="medium")

        with l:
            render_alerts(st.session_state.initial_alerts, scenario.get("noise_alerts",[]))
            render_services(obs.visible_services, obs.dependencies, res["root_cause"])

        with m:
            render_action_trace(st.session_state.steps_log)
            if show_trace:
                render_reasoning(st.session_state.reasoning_log, st.session_state.steps_log, res["root_cause"], res["agent_answer"])

        with r:
            render_causal_graph(obs.dependencies, res["root_cause"], show_chain)
            render_logs(obs.logs, obs.dependencies)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:0 28px">', unsafe_allow_html=True)
        render_empty()
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()