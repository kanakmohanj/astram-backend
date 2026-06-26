# backend/app/services/agent_engine/nodes.py

import operator
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from .prompts import ORCHESTRATOR_SYSTEM_PROMPT
from ..rag_memory.retriever import memory_bank
from ..ml_forecasting.model import forecaster_instance
from ..ml_forecasting.feature_eng import calculate_impact_score
from ..graph_routing.router import DynamicRouter
from ..graph_routing.graph_builder import graph_builder

# Instantiate the router globally so the graph index is built only once
global_router = DynamicRouter(graph_builder.graph)

# Define the State our Graph will pass between nodes
class AgentState(TypedDict):
    event_details: Dict[str, Any]
    predicted_duration: float
    impact_score: float
    diversion_routes: List[Dict[str, Any]]
    historical_cases: List[Dict[str, Any]]
    resource_requirements: Dict[str, int]
    # Annotated with operator.add tells LangGraph to append instead of overwrite
    reasoning_chain: Annotated[List[Dict[str, Any]], operator.add] 
    drafted_plan: str

def ingest_data_node(state: AgentState) -> AgentState:
    """Node 1: Runs ML Forecaster and Graph Routing"""
    event = state['event_details']
    
    # 1. Predict Duration
    pred_duration = forecaster_instance.predict_duration(event) 
    
    # 2. Calculate Impact Score
    impact = calculate_impact_score(event.get('event_cause', ''), event.get('priority', ''), event.get('hour', 0), pred_duration)
    
    # 3. Calculate Diversions
    start_junc = event.get('junction')
    end_junc = event.get('end_junction', start_junc) # Fallback to start if missing
    
    routes = global_router.calculate_diversion(
        event_corridor=event.get('corridor'), 
        impact_score=impact, 
        start=start_junc, 
        end=end_junc
    )
    
    diversion_options = routes.get("diversion_options", [])
    
    # Log the exact math for the UI
    log = {
        "step": "Data Ingestion & ML Forecasting",
        "action": f"Ran XGBoost. Predicted duration: {pred_duration:.1f} mins. Calculated Impact: {impact}%",
        "data_used": {"event": event.get('event_cause'), "priority": event.get('priority')}
    }
    
    return {
        "predicted_duration": pred_duration,
        "impact_score": impact,
        "diversion_routes": diversion_options,
        # Because of Annotated[..., operator.add], we just return the new list item
        "reasoning_chain": [log] 
    }

def logic_resource_node(state: AgentState) -> AgentState:
    """Node 2: Rule-based Resource Calculation"""
    impact = state['impact_score']
    resources = {"inspectors": 0, "constables": 0, "barricades": 0, "towing_vehicles": 0}
    
    if impact >= 80:
        resources.update({"inspectors": 1, "constables": 6, "barricades": 30, "towing_vehicles": 1})
    elif impact >= 50:
        resources.update({"inspectors": 0, "constables": 3, "barricades": 15, "towing_vehicles": 1})
    else:
        resources.update({"inspectors": 0, "constables": 1, "barricades": 5, "towing_vehicles": 0})
        
    log = {
        "step": "Resource Calculation",
        "action": f"Applied rule-based logic for impact score {impact}%",
        "data_used": resources
    }
    
    return {
        "resource_requirements": resources,
        "reasoning_chain": [log]
    }

def historical_context_node(state: AgentState) -> AgentState:
    """Node 3: Fetch past similar events from Pinecone Vector DB"""
    event = state['event_details']
    
    # Query the Cloud Vector Database for real past events
    past_events = memory_bank.retrieve_similar_past_events(
        current_cause=event.get('event_cause', ''),
        current_location=event.get('corridor', event.get('junction', '')),
        k=2 
    )
    
    log = {
        "step": "Historical Data Retrieval (RAG)",
        "action": f"Found {len(past_events)} highly-rated past operations for context.",
        "data_used": past_events
    }
    
    return {
        "historical_cases": past_events,
        "reasoning_chain": [log]
    }

def llm_planner_node(state: AgentState) -> AgentState:
    """Node 4: Generates the natural language brief"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    # Safely extract the top route, checking for the single-node / no-path error dict
    top_route_info = "None available"
    if state['diversion_routes']:
        top_route = state['diversion_routes'][0]
        if "error" in top_route:
             top_route_info = f"Routing Error/Not Applicable: {top_route['error']}"
        else:
             top_route_info = str(top_route)
             
    user_content = f"""
    Event Details: {state['event_details']}
    Predicted Duration: {state['predicted_duration']} minutes
    Impact Score: {state['impact_score']}%
    Required Resources: {state['resource_requirements']}
    Top Diversion Route: {top_route_info}
    Past Similar Event Data: {state['historical_cases']}
    """
    
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    
    log = {
        "step": "LLM Synthesis",
        "action": "Generated natural language brief using GPT-4o.",
        "data_used": {"tokens_used": response.response_metadata.get('token_usage', 'unknown')}
    }
    
    return {
        "drafted_plan": response.content,
        "reasoning_chain": [log]
    }