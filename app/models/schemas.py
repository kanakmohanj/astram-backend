from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class TrafficEventBase(BaseModel):
    id: str
    event_cause: str
    latitude: float
    longitude: float
    corridor: str
    priority: str
    junction: str
    end_junction: Optional[str] = None
    hour: int

class FeedbackPayload(BaseModel):
    event_id: str
    event_cause: str
    location: str
    ai_plan: str
    officer_feedback: str
    rating: int

class AIOrchestrationResponse(BaseModel):
    predicted_duration: float
    impact_score: float
    diversion_routes: List[Dict[str, Any]]
    resource_requirements: Dict[str, int]
    reasoning_chain: List[Dict[str, Any]]
    drafted_plan: str