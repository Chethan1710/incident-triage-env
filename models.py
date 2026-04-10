from pydantic import BaseModel
from typing import List, Dict, Optional


class Observation(BaseModel):
    alerts: List[Dict]
    logs: List[str]
    visible_services: List[str]
    dependencies: Dict[str, List[str]]
    history: List[Dict]


class Action(BaseModel):
    action_type: str
    target: Optional[str] = None


class Reward(BaseModel):
    value: float
