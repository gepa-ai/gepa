"""Initial search algorithm for broadcast path optimization.

This module contains the baseline search_algorithm that will be evolved
by GEPA to find better broadcast routing strategies.
"""

import networkx as nx
import pandas as pd
import os
from typing import Dict, List

from examples.adrs.cloudcast.utils.cloudcast.broadcast import BroadCastTopology


# EVOLVE-BLOCK-START
def search_algorithm(src, dsts, G, num_partitions):
    """
    Find broadcast paths from source to all destinations.
    
    This is the function that GEPA will evolve to optimize broadcast routing.
    The baseline implementation uses Dijkstra's shortest path algorithm
    based on cost as the edge weight.
    
    Args:
        src: Source node identifier (e.g., "aws:ap-northeast-1")
        dsts: List of destination node identifiers
        G: NetworkX DiGraph with cost and throughput edge attributes
        num_partitions: Number of data partitions
        
    Returns:
        BroadCastTopology object with paths for all destinations and partitions
    """
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc_topology = BroadCastTopology(src, dsts, num_partitions)

    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(0, len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(bc_topology.num_partitions):
                bc_topology.append_dst_partition_path(dst, j, [s, t, G[s][t]])

    return bc_topology
# EVOLVE-BLOCK-END


def make_nx_graph(cost_path=None, throughput_path=None, num_vms=1):
    """
    Create a NetworkX graph with capacity constraints and cost info.
    
    This is included in the initial program so evolved programs have access
    to the graph creation utility.
    
    Args:
        cost_path: Path to cost CSV file
        throughput_path: Path to throughput CSV file  
        num_vms: Number of VMs per region (scales throughput)
        
    Returns:
        networkx.DiGraph with cost and throughput edge attributes
    """
    # Get paths relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cost_path is None:
        cost_path = os.path.join(current_dir, "profiles/cost.csv")
    
    if throughput_path is None:
        throughput_path = os.path.join(current_dir, "profiles/throughput.csv")
    
    cost = pd.read_csv(cost_path)
    throughput = pd.read_csv(throughput_path)

    G = nx.DiGraph()
    for _, row in throughput.iterrows():
        if row["src_region"] == row["dst_region"]:
            continue
        G.add_edge(
            row["src_region"], 
            row["dst_region"], 
            cost=None, 
            throughput=num_vms * row["throughput_sent"] / 1e9
        )

    for _, row in cost.iterrows():
        if row["src"] in G and row["dest"] in G[row["src"]]:
            G[row["src"]][row["dest"]]["cost"] = row["cost"]

    # some pairs not in the cost grid
    no_cost_pairs = []
    for edge in G.edges.data():
        src, dst = edge[0], edge[1]
        if edge[-1]["cost"] is None:
            no_cost_pairs.append((src, dst))
    
    if no_cost_pairs:
        print("Unable to get costs for: ", no_cost_pairs)

    return G


# Helper functions that won't be evolved
def create_broadcast_topology(src: str, dsts: List[str], num_partitions: int = 4):
    """Create a broadcast topology instance."""
    return BroadCastTopology(src, dsts, num_partitions)


def run_search_algorithm(src: str, dsts: List[str], G, num_partitions: int):
    """Run the search algorithm and return the topology."""
    return search_algorithm(src, dsts, G, num_partitions)
