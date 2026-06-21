from fastapi import APIRouter
from .routes import events

api_router = APIRouter()
# We attach the events router (which contains our websocket and AI triggers from Phase 5)
api_router.include_router(events.router, tags=["events"])