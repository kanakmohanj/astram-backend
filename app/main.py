# app/main.py

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
    title="Astram Orchestrator API",
    description="Agentic Decision Support System for Traffic Management",
    version="1.0.0"
)

# Pull the frontend URL from the environment, defaulting to localhost for dev
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Allow both the explicit Frontend URL and localhost (for local testing)
origins = list(set([FRONTEND_URL, "http://localhost:5173"]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the consolidated API router
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"status": "Astram Orchestrator Backend is Running!"}