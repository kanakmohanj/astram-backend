# backend/app/api/routes/events.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from typing import List
import json
import hashlib
from datetime import datetime, timezone
from cachetools import TTLCache

# Import our ML, Agent, and Memory services
from ...services.ml_forecasting.feature_eng import calculate_impact_score
from ...services.agent_engine.workflow import orchestrator_app
from ...services.rag_memory.retriever import memory_bank
from ...models.schemas import FeedbackPayload # Ensure this is in your schemas.py

router = APIRouter()

# --- 1. WEBSOCKET MANAGER (For Live UI Updates) ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Failed to send to a websocket client: {e}")

manager = ConnectionManager()

@router.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- 2. CACHE SETUP (Short-Term Tactical Memory) ---
ai_deployment_cache = TTLCache(maxsize=100, ttl=900)

def generate_cache_key(event_data: dict) -> str:
    signature = {
        "cause": event_data.get('event_cause'),
        "corridor": event_data.get('corridor'),
        "hour": event_data.get('hour')
    }
    return hashlib.md5(json.dumps(signature, sort_keys=True).encode()).hexdigest()

# --- 3. INGESTION ENDPOINT (Field -> Backend -> React) ---
@router.post("/api/v1/ingest-event")
async def ingest_new_event(event_data: dict, background_tasks: BackgroundTasks):
    base_impact = calculate_impact_score(
        event_cause=event_data.get('event_cause', 'others'),
        priority=event_data.get('priority', 'Low'),
        hour=datetime.now(timezone.utc).hour,
        predicted_duration=60.0 
    )
    
    event_data['status'] = 'active'
    event_data['initial_impact_score'] = base_impact
    event_data['ingested_at'] = datetime.now(timezone.utc).isoformat()
    
    # Broadcast to React instantly without blocking the HTTP response
    background_tasks.add_task(manager.broadcast, {
        "type": "NEW_EVENT",
        "payload": event_data
    })
    
    return {"status": "Ingested successfully", "event_id": event_data.get('id')}

# --- 4. AI ORCHESTRATOR ENDPOINT (Admin Clicks UI) ---
@router.post("/api/v1/events/{event_id}/orchestrate")
async def run_ai_orchestrator(event_id: str, event_data: dict):
    cache_key = generate_cache_key(event_data)
    
    if cache_key in ai_deployment_cache:
        cached_result = ai_deployment_cache[cache_key]
        response = dict(cached_result)
        response['reasoning_chain'] = response.get('reasoning_chain', []) + [{
            "step": "System Optimization",
            "action": "Retrieved from High-Speed Edge Cache.",
            "details": {"cache_ttl": "15 minutes"}
        }]
        return response

    initial_state = {
        "event_details": event_data,
        "reasoning_chain": []
    }
    
    # CRITICAL FIX: Asynchronous invocation to prevent server freezing
    final_state = await orchestrator_app.ainvoke(initial_state)
    
    ai_deployment_cache[cache_key] = final_state
    
    return final_state

# --- 5. RAG FEEDBACK ENDPOINT (The Learning Loop) ---
# Standardized to /api/v1/...
@router.post("/api/v1/events/feedback")
async def submit_event_feedback(payload: FeedbackPayload):
    """
    Receives post-event feedback from the React frontend.
    If rating >= 4, embeds the successful tactic into Pinecone.
    """
    if payload.rating < 1 or payload.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5.")

    result = memory_bank.learn_from_feedback(
        event_id=payload.event_id,
        event_cause=payload.event_cause,
        location=payload.location,
        ai_plan=payload.ai_plan,
        officer_feedback=payload.officer_feedback,
        rating=payload.rating
    )
    
    return result