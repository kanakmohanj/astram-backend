import networkx as nx
import pandas as pd
import math

class TrafficGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph() 
        self.AVG_SPEED_KMH = 30.0 

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates distance between two GPS coordinates in kilometers."""
        R = 6371.0 
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _spatially_sort_junctions(self, junctions: list) -> list:
        """
        Sorts a list of junctions geometrically so they form a continuous line.
        """
        if len(junctions) <= 2:
            return junctions
            
        # 1. Find the two junctions furthest apart (the endpoints of the corridor)
        max_dist = -1
        endpoints = (junctions[0], junctions[1])
        
        for i in range(len(junctions)):
            for j in range(i + 1, len(junctions)):
                lat1, lon1 = self.graph.nodes[junctions[i]]['pos']
                lat2, lon2 = self.graph.nodes[junctions[j]]['pos']
                dist = self._haversine(lat1, lon1, lat2, lon2)
                if dist > max_dist:
                    max_dist = dist
                    endpoints = (junctions[i], junctions[j])
                    
        # 2. Start from one endpoint and build a nearest-neighbor chain
        start_node = endpoints[0]
        unvisited = set(junctions)
        unvisited.remove(start_node)
        
        sorted_chain = [start_node]
        current_node = start_node
        
        while unvisited:
            lat_c, lon_c = self.graph.nodes[current_node]['pos']
            
            # Find the closest unvisited neighbor
            closest_node = None
            min_dist = float('inf')
            
            for candidate in unvisited:
                lat_cand, lon_cand = self.graph.nodes[candidate]['pos']
                dist = self._haversine(lat_c, lon_c, lat_cand, lon_cand)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = candidate
                    
            sorted_chain.append(closest_node)
            unvisited.remove(closest_node)
            current_node = closest_node
            
        return sorted_chain

    def build_graph_from_data(self, df: pd.DataFrame) -> nx.Graph:
        print("Building Base Traffic Graph...")
        
        valid_data = df.dropna(subset=['junction', 'corridor', 'latitude', 'longitude']).copy()
        
        nodes_df = valid_data.groupby('junction').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()

        for _, row in nodes_df.iterrows():
            self.graph.add_node(
                row['junction'], 
                pos=(row['latitude'], row['longitude'])
            )

        corridors = valid_data.groupby('corridor')['junction'].unique()
        
        for corridor_name, junctions in corridors.items():
            junctions_list = list(junctions)
            
            # Apply spatial sorting to prevent the "Spaghetti Graph"
            sorted_junctions = self._spatially_sort_junctions(junctions_list)
            
            for i in range(len(sorted_junctions) - 1):
                j1 = sorted_junctions[i]
                j2 = sorted_junctions[i+1]
                
                lat1, lon1 = self.graph.nodes[j1]['pos']
                lat2, lon2 = self.graph.nodes[j2]['pos']
                
                dist_km = self._haversine(lat1, lon1, lat2, lon2)
                base_time_mins = max(1.0, (dist_km / self.AVG_SPEED_KMH) * 60.0)
                
                self.graph.add_edge(
                    j1, j2, 
                    corridor=corridor_name, 
                    distance_km=round(dist_km, 2),
                    base_weight=round(base_time_mins, 2),
                    current_weight=round(base_time_mins, 2)
                )
                
        print(f"Graph built with {self.graph.number_of_nodes()} junctions and {self.graph.number_of_edges()} corridors.")
        return self.graph

graph_builder = TrafficGraphBuilder()

# # backend/app/services/graph_routing/graph_builder.py

# import networkx as nx
# import pandas as pd
# import numpy as np
# import math
# from typing import Dict, Any, Tuple

# class TrafficGraphBuilder:
#     def __init__(self):
#         self.graph = nx.Graph()
#         # Average city speed in km/h to calculate base travel time
#         self.AVG_SPEED_KMH = 30.0 

#     def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#         """Calculates distance between two GPS coordinates in kilometers."""
#         R = 6371.0 # Earth radius in km
        
#         dlat = math.radians(lat2 - lat1)
#         dlon = math.radians(lon2 - lon1)
#         a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
#         c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#         return R * c

#     def build_graph_from_data(self, df: pd.DataFrame) -> nx.Graph:
#         """
#         Constructs the base topological graph from historical event data.
#         Nodes = Junctions, Edges = Corridors.
#         """
#         print("Building Base Traffic Graph...")
        
#         # Clean data for valid locations
#         valid_data = df.dropna(subset=['junction', 'corridor', 'latitude', 'longitude']).copy()
        
#         # Extract unique nodes (junctions) and their average coordinates
#         nodes_df = valid_data.groupby('junction').agg({
#             'latitude': 'mean',
#             'longitude': 'mean'
#         }).reset_index()

#         # Add nodes to graph
#         for _, row in nodes_df.iterrows():
#             self.graph.add_node(
#                 row['junction'], 
#                 pos=(row['latitude'], row['longitude'])
#             )

#         # To build edges (corridors), we group by corridor and connect junctions on that corridor
#         corridors = valid_data.groupby('corridor')['junction'].unique()
        
#         for corridor_name, junctions in corridors.items():
#             # If a corridor has multiple junctions, connect them sequentially or fully
#             # For this simulation, we connect sequentially based on distance or order
#             junctions_list = list(junctions)
#             for i in range(len(junctions_list) - 1):
#                 j1 = junctions_list[i]
#                 j2 = junctions_list[i+1]
                
#                 # Get coordinates
#                 lat1, lon1 = self.graph.nodes[j1]['pos']
#                 lat2, lon2 = self.graph.nodes[j2]['pos']
                
#                 # Calculate base weight (travel time in minutes)
#                 dist_km = self._haversine(lat1, lon1, lat2, lon2)
#                 base_time_mins = (dist_km / self.AVG_SPEED_KMH) * 60.0
                
#                 # Ensure minimum 1 minute travel time to prevent 0-weight edge errors
#                 base_time_mins = max(1.0, base_time_mins)
                
#                 # Add edge with metadata
#                 self.graph.add_edge(
#                     j1, j2, 
#                     corridor=corridor_name, 
#                     distance_km=round(dist_km, 2),
#                     base_weight=round(base_time_mins, 2),
#                     current_weight=round(base_time_mins, 2) # Will be dynamically updated
#                 )
                
#         print(f"Graph built with {self.graph.number_of_nodes()} junctions and {self.graph.number_of_edges()} corridors.")
#         return self.graph

# # Singleton instance
# graph_builder = TrafficGraphBuilder()