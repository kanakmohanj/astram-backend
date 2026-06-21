# backend/app/services/agent_engine/workflow.py

from langgraph.graph import StateGraph, END
from .nodes import (
    AgentState, 
    ingest_data_node, 
    logic_resource_node, 
    historical_context_node, 
    llm_planner_node
)

def create_orchestrator_graph() -> StateGraph:
    """
    Assembles the multi-agent workflow.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Data_Ingestion", ingest_data_node)
    workflow.add_node("Resource_Logic", logic_resource_node)
    workflow.add_node("Historical_RAG", historical_context_node)
    workflow.add_node("LLM_Planner", llm_planner_node)
    
    # Define the execution edges
    workflow.add_edge("Data_Ingestion", "Resource_Logic")
    workflow.add_edge("Resource_Logic", "Historical_RAG")
    workflow.add_edge("Historical_RAG", "LLM_Planner")
    workflow.add_edge("LLM_Planner", END)
    
    workflow.set_entry_point("Data_Ingestion")
    
    return workflow.compile()

# Singleton instance for the FastAPI router
orchestrator_app = create_orchestrator_graph()

# --- Async FastAPI endpoint runner ---
async def run_agent(event_data: dict):
    # Only seed the data we have. LangGraph handles the rest.
    initial_state = {
        "event_details": event_data,
        "reasoning_chain": []
    }
    
    # CRITICAL: Use await and ainvoke for non-blocking execution
    final_state = await orchestrator_app.ainvoke(initial_state)
    
    return final_state