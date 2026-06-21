# backend/app/services/graph_routing/router.py

import networkx as nx
from typing import List, Dict, Any
from itertools import islice

class DynamicRouter:
    def __init__(self, base_graph: nx.Graph):
        self.base_graph = base_graph.copy()
        
        # Pre-compute an index of corridor -> list of edges for O(1) lookups
        self.corridor_edges = {}
        for u, v, data in self.base_graph.edges(data=True):
            corr = data.get('corridor')
            if corr:
                self.corridor_edges.setdefault(corr, []).append((u, v))

    def apply_event_penalty(self, event_corridor: str, impact_score: float) -> nx.Graph:
        working_graph = self.base_graph.copy()
        PENALTY_MULTIPLIER = 5.0 
        
        penalty_factor = 1.0 + ((impact_score / 100.0) * PENALTY_MULTIPLIER)
        
        # O(1) lookup to instantly find only the affected edges
        edges_to_penalize = self.corridor_edges.get(event_corridor, [])
        
        for u, v in edges_to_penalize:
            base_w = working_graph[u][v]['base_weight']
            new_weight = base_w * penalty_factor
            working_graph[u][v]['current_weight'] = round(new_weight, 2)
                
        return working_graph

    def get_k_alternative_routes(self, 
                               working_graph: nx.Graph, 
                               source_junction: str, 
                               target_junction: str, 
                               k: int = 3) -> List[Dict[str, Any]]:
        """
        Implements Yen's K-Shortest Paths to find multiple diversion routes.
        Returns the top K routes with their total expected travel times.
        """
        if source_junction not in working_graph or target_junction not in working_graph:
            return [{"error": "Source or Target junction not found in graph."}]

        routes = []
        
        try:
            # nx.shortest_simple_paths acts as Yen's Algorithm when evaluated lazily via islice
            k_paths_generator = nx.shortest_simple_paths(
                working_graph, 
                source=source_junction, 
                target=target_junction, 
                weight='current_weight'
            )
            
            for path in islice(k_paths_generator, k):
                # Calculate total dynamic time and distance for this specific route
                total_time = 0.0
                total_distance = 0.0
                corridors_used = []
                
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = working_graph[u][v]
                    total_time += edge_data['current_weight']
                    total_distance += edge_data['distance_km']
                    
                    if edge_data['corridor'] not in corridors_used:
                        corridors_used.append(edge_data['corridor'])

                routes.append({
                    "route_path": path, # List of junctions
                    "corridors_to_take": corridors_used,
                    "estimated_time_mins": round(total_time, 2),
                    "total_distance_km": round(total_distance, 2)
                })
                
        except nx.NetworkXNoPath:
            return [{"error": "No valid route exists between these junctions."}]
            
        return routes

    def calculate_diversion(self, event_corridor: str, impact_score: float, start: str, end: str) -> dict:
        """
        Master function to be called by the FastAPI Endpoint.
        """
        # 1. Apply the mathematical penalty to the graph
        penalized_graph = self.apply_event_penalty(event_corridor, impact_score)
        
        # 2. Run Yen's algorithm to get top 3 alternatives
        alternatives = self.get_k_alternative_routes(penalized_graph, start, end, k=3)
        
        return {
            "event_corridor": event_corridor,
            "applied_impact_score": impact_score,
            "diversion_options": alternatives
        }