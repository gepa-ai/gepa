"""Broadcast topology data structures for cloud data transfer optimization."""

import networkx as nx
import json
from typing import Dict, List


class SingleDstPath(Dict):
    partition: int
    edges: List[List]  # [[src, dst, edge data]]


class BroadCastTopology:
    """
    Represents a broadcast topology for multi-destination data transfer.
    
    The topology tracks paths from a single source to multiple destinations,
    with support for data partitioning to optimize transfer efficiency.
    """
    
    def __init__(
        self, 
        src: str, 
        dsts: List[str], 
        num_partitions: int = 4, 
        paths: Dict[str, SingleDstPath] = None
    ):
        self.src = src  # single str
        self.dsts = dsts  # list of strs
        self.num_partitions = num_partitions

        # dict(dst) --> dict(partition) --> list(nx.edges)
        # example: {dst1: {partition1: [src->node1, node1->dst1], partition 2: [src->dst1]}}
        if paths is not None:
            self.paths = paths
            self.set_graph()
        else:
            self.paths = {
                dst: {str(i): None for i in range(num_partitions)} 
                for dst in dsts
            }

    def get_paths(self):
        """Return all paths in the topology."""
        return self.paths

    def set_num_partitions(self, num_partitions: int):
        """Set the number of data partitions."""
        self.num_partitions = num_partitions

    def set_dst_partition_paths(self, dst: str, partition: int, paths: List[List]):
        """
        Set paths for partition = partition to reach dst.
        """
        partition = str(partition)
        self.paths[dst][partition] = paths

    def append_dst_partition_path(self, dst: str, partition: int, path: List):
        """
        Append path for partition = partition to reach dst.
        """
        partition = str(partition)
        if self.paths[dst][partition] is None:
            self.paths[dst][partition] = []
        self.paths[dst][partition].append(path)

    def set_graph(self):
        """Initialize graph from paths (placeholder for extension)."""
        pass
