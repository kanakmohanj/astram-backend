# 🚦 ASTRAM: AI-Powered Traffic Orchestration Engine

**ASTRAM** (Automated System for Traffic Response and Management) is an enterprise-grade, agentic AI backend designed to predict, manage, and orchestrate responses to real-time traffic incidents.

Built for the Flipkart Hackathon, ASTRAM moves beyond basic CRUD applications by combining deterministic Machine Learning forecasting with Generative AI and continuous learning (RAG). It dynamically drafts highly contextual tactical plans for on-ground traffic officers, creating a closed-loop system that learns from past operational successes.

---

## 📖 Executive Summary & Core Pipeline

The backend is built on **FastAPI** to ensure high-concurrency asynchronous processing. The core orchestration is powered by **LangGraph**, which routes incidents through a multi-agent pipeline:

1. **⚡ Real-Time Ingestion:** Secure REST endpoints and WebSockets capable of ingesting high-frequency incident data from IoT cameras or manual reports.
2. **📈 Predictive ML Engine:** An XGBoost regressor loaded directly into RAM evaluates 13 spatial-temporal features (e.g., rush hour, weekend, priority score) to instantly forecast incident clearance duration.
3. **⚙️ Rule-Based Resource Allocation:** Deterministic algorithms calculate precise physical resource requirements (constables, towing vehicles, barricades) based on the ML impact score.
4. **🧠 Continuous Learning Memory (RAG):** Utilizes a local FAISS Vector Database to store historical incidents that received a 4/5 or 5/5 success rating. It uses `all-mpnet-base-v2` embeddings to perform semantic similarity searches, giving the AI "memory" of past operations.
5. **🤖 LLM Tactical Synthesis:** Google Gemini 1.5 Pro ingests the ML predictions, resource logic, and historical RAG context to draft a natural language, actionable brief for dispatchers.

---

## 🛠️ Tech Stack

- **Framework:** FastAPI, Uvicorn
- **AI Orchestration:** LangGraph, LangChain
- **Generative AI:** Google Gemini 1.5 Pro
- **Machine Learning:** XGBoost, Pandas
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** HuggingFace (`all-mpnet-base-v2`)

---

## 🏗️ Engineering Challenges & Architectural Achievements

During development, we designed the system to be highly resilient against real-world deployment challenges:

- **Challenge 1: HuggingFace API Throttling & Timeouts**
  - _The Problem:_ The RAG pipeline initially attempted to ping the HuggingFace API to verify embedding model versions on every request, causing 15+ second timeouts.
  - _The Fix:_ We migrated to **100% Offline Local Embeddings** by caching the 438MB `all-mpnet-base-v2` model directly into the server environment and forcing the `HF_HUB_OFFLINE="1"` variable.
  - _Result:_ Zero-latency text embedding operations. The system is no longer reliant on external API uptime.

- **Challenge 2: Enterprise Network Blocks on Cloud Databases**
  - _The Problem:_ Strict institutional firewalls actively blocked outbound connections to cloud vector databases, crashing the LangGraph pipeline entirely.
  - _The Fix:_ We implemented a **100% Local Vector Database (FAISS)** and wrapped connections in graceful degradation blocks. If a network drop occurs, the AI catches the error, bypasses the RAG step, and successfully generates a baseline tactical plan without crashing.
  - _Result:_ Guaranteed 100% uptime for the core AI orchestrator, even in highly restricted or unstable network environments.

- **Challenge 3: Unpredictable LLM JSON Outputs**
  - _The Problem:_ Generative models occasionally return unstructured data, causing frontends to crash during re-renders.
  - _The Fix:_ Standardized Pydantic schemas in FastAPI to enforce strict output validation before data ever leaves the backend.

---

## 🚀 Local Setup & Installation

Follow these steps to run the ASTRAM backend locally.

### Prerequisites

- Python 3.10 or higher
- Git

### Step 1: Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/astram-backend.git](https://github.com/YOUR_USERNAME/astram-backend.git)
cd astram-backend
```

### Step 2: Create a Virtual Environment

# Windows

python -m venv .venv
.venv\Scripts\activate

# Mac/Linux

python3 -m venv .venv
source .venv/bin/activate

### Step 3: Install Dependencies

pip install fastapi uvicorn xgboost pandas langchain langgraph google-generativeai faiss-cpu langchain-community langchain-huggingface sentence-transformers

### Step 4: Configure Environment Variables

Create a file named .env in the root directory of the project and add the following:

# Get this from: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

GEMINI_API_KEY="your-gemini-api-key-here"

# Current Environment Status

ENVIRONMENT="development"

# CRITICAL: Forces HuggingFace to use local cache. Prevents timeout errors.

HF_HUB_OFFLINE="1"

### Step 5: Run the Server

uvicorn app.main:app --reload

The backend will boot up, load the ML models into RAM, and expose the API at http://127.0.0.1:8000.

### 📡 API Endpoints Overview

View the interactive API documentation (Swagger UI) at: 👉 http://127.0.0.1:8000/docs

POST /api/v1/events/ingest - Ingest a new traffic incident from an external device or UI.

POST /api/v1/events/{event_id}/orchestrate - Trigger the LangGraph AI pipeline.

POST /api/v1/events/feedback - Submit 1-5 star ratings to train the RAG memory.

WS /ws/dashboard - Live WebSocket feed for real-time frontend updates.

### 🔮 Future Roadmap

Interactive Geo-Spatial Command Center: Integrating Leaflet.js into the React dashboard to map live incidents based on WebSocket coordinates.

Omnichannel Ingestion: Expanding the /ingest webhook to accept automated alerts from Google Maps/Waze traffic APIs.

Automated Cloud Syncing: Implementing a cron job to sync the local FAISS database with a centralized cloud data warehouse for cross-city learning.

Built for the Flipkart Hackathon 🚀
