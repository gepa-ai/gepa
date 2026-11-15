"""Broadcast simulator for evaluating cloud data transfer paths."""

from typing import List
from pprint import pprint
import networkx as nx
import json

from examples.adrs.cloudcast.utils.cloudcast.broadcast import BroadCastTopology

# Optional dependency for colored output
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Define dummy color codes if colorama not available
    class Fore:
        BLUE = ""
        YELLOW = ""
        RED = ""
        GREEN = ""
    class Style:
        RESET_ALL = ""


class BCSimulator:
    """
    Broadcast Simulator for evaluating multi-destination data transfer paths.
    
    Evaluates the cost and transfer time of broadcast topologies by simulating
    data flow through the network while respecting bandwidth constraints.
    """
    
    # Default variables
    data_vol: float = 4.0  # size of data to be sent to multiple dsts (GB)
    num_partitions: int = 1
    partition_data_vol: int = data_vol / num_partitions
    default_vms_per_region: int = 1
    cost_per_instance_hr: float = 0.54  # based on m5.8xlarge spot
    src: str
    dsts: List[str]
    algo: str
    g = nx.DiGraph

    def __init__(self, num_vms, output_dir=None):
        """
        Initialize the simulator.
        
        Args:
            num_vms: Number of VMs per region
            output_dir: Directory to write output files
        """
        self.output_dir = output_dir
        self.default_vms_per_region = num_vms

    def initialization(self, path, config):
        """
        Initialize simulation from path data and configuration.
        
        Args:
            path: Either a file path to JSON or a BroadCastTopology object
            config: Configuration dict with data_vol, num_partitions, etc.
        """
        # check if path is dict
        if isinstance(path, str):
            # Read from json
            with open(path, "r") as f:
                data = json.loads(f.read())
        else:
            data = {
                "algo": "none",
                "source_node": path.src,
                "terminal_nodes": path.dsts,
                "num_partitions": path.num_partitions,
                "generated_path": path.paths,
            }

        self.src = data["source_node"]
        self.dsts = data["terminal_nodes"]
        self.algo = data["algo"]
        self.paths = data["generated_path"]

        self.num_partitions = config["num_partitions"]
        self.data_vol = config["data_vol"]
        self.partition_data_vol = self.data_vol / self.num_partitions

        # Default in/egress limit if not set
        providers = ["aws", "gcp", "azure"]
        provider_ingress = [10, 16, 16]
        provider_egress = [5, 7, 16]
        self.ingress_limits = {providers[i]: provider_ingress[i] for i in range(len(providers))}
        self.egress_limits = {providers[i]: provider_egress[i] for i in range(len(providers))}

        if "ingress_limit" in config:
            for p, limit in config["ingress_limit"].items():
                self.ingress_limits[p] = self.default_vms_per_region * limit

        if "egress_limit" in config:
            for p, limit in config["egress_limit"].items():
                self.egress_limits[p] = self.default_vms_per_region * limit

    def evaluate_path(self, path, config, write_to_file=False):
        """
        Evaluate a broadcast path configuration.
        
        Args:
            path: Path data (file path or BroadCastTopology)
            config: Configuration dict
            write_to_file: Whether to write results to output_dir
            
        Returns:
            Tuple of (max_transfer_time, total_cost)
        """
        self.initialization(path, config)

        # construct graph
        self.g = self.__construct_g()

        # evaluate transfer time and total cost
        max_t, avg_t, last_dst = self.__transfer_time()
        self.cost = self.__total_cost()

        # output to json file
        if write_to_file and self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            with open(f"{self.output_dir}/{self.algo}_eval.json", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "path": path if isinstance(path, str) else "topology_object",
                            "max_transfer_time": max_t,
                            "avg_transfer_time": avg_t,
                            "last_dst": last_dst,
                            "tot_cost": self.cost,
                        }
                    )
                )
        return max_t, self.cost

    def __construct_g(self):
        """Construct a graph based on the given topology."""
        g = nx.DiGraph()
        for dst in self.dsts:
            for partition_id in range(self.num_partitions):
                for edge in self.paths[dst][str(partition_id)]:
                    src, dst_node, edge_data = edge[0], edge[1], edge[2]
                    if not g.has_edge(src, dst_node):
                        cost = edge_data["cost"]
                        throughput = edge_data["throughput"]
                        g.add_edge(
                            src, dst_node, 
                            throughput=throughput, 
                            cost=edge_data["cost"], 
                            flow=throughput
                        )
                        g[src][dst_node]["partitions"] = set()
                    g[src][dst_node]["partitions"].add(partition_id)

        # Proportionally share if exceed in/egress limit of any node
        for node in g.nodes:
            provider = node.split(":")[0]

            in_edges, out_edges = g.in_edges(node), g.out_edges(node)
            in_flow_sum = sum([g[i[0]][i[1]]["flow"] for i in in_edges])
            out_flow_sum = sum([g[o[0]][o[1]]["flow"] for o in out_edges])

            if in_flow_sum > self.ingress_limits.get(provider, 10):
                for edge in in_edges:
                    src, dst = edge[0], edge[1]
                    flow_proportion = 1 / len(list(in_edges))
                    g[src][dst]["flow"] = min(
                        g[src][dst]["flow"], 
                        self.ingress_limits.get(provider, 10) * flow_proportion
                    )

            if out_flow_sum > self.egress_limits.get(provider, 5):
                for edge in out_edges:
                    src, dst = edge[0], edge[1]
                    flow_proportion = 1 / len(list(out_edges))
                    g[src][dst]["flow"] = min(
                        g[src][dst]["flow"], 
                        self.egress_limits.get(provider, 5) * flow_proportion
                    )

        return g

    def __get_path(self):
        """Get all simple paths from source to destinations."""
        all_paths = [
            path for node in self.dsts 
            for path in nx.all_simple_paths(self.g, self.src, node)
        ]
        return all_paths

    def __slowest_capacity_link(self):
        """Find the minimum throughput link in the graph."""
        min_tput = min([edge[-1]["throughput"] for edge in self.g.edges().data()])
        return min_tput

    def __transfer_time(self, log=True):
        """
        Calculate transfer time for each destination.
        
        Returns:
            Tuple of (max_time, avg_time, last_destinations)
        """
        t_dict = dict()
        for dst in self.dsts:
            partition_time = float("-inf")
            for i in range(self.num_partitions):
                for edge in self.paths[dst][str(i)]:
                    edge_data = self.g[edge[0]][edge[1]]
                    partition_time = max(
                        partition_time, 
                        len(edge_data["partitions"]) * self.partition_data_vol * 8 / edge_data["flow"]
                    )
            t_dict[dst] = partition_time

        max_t = max(t_dict.values())
        last_dst = [k for k, v in t_dict.items() if v == max_t]
        avg_t = sum(t_dict.values()) / len(t_dict.values())
        
        return max_t, avg_t, last_dst

    def __total_cost(self):
        """Calculate total cost (egress + instance costs)."""
        sum_egress_cost = 0
        for edge in self.g.edges.data():
            edge_data = edge[-1]
            sum_egress_cost += (
                len(edge_data["partitions"]) * self.partition_data_vol * edge_data["cost"]
            )

        runtime_s, _, _ = self.__transfer_time(log=False)
        runtime_s = round(runtime_s, 2)
        sum_instance_cost = 0
        for node in self.g.nodes():
            sum_instance_cost += (
                self.default_vms_per_region * (self.cost_per_instance_hr / 3600) * runtime_s
            )

        sum_cost = sum_egress_cost + sum_instance_cost
        print(
            f"{Fore.BLUE}Sum of total cost = egress cost {Fore.YELLOW}(${round(sum_egress_cost, 4)}) "
            f"{Fore.BLUE}+ instance cost {Fore.YELLOW}(${round(sum_instance_cost, 4)}) "
            f"{Fore.BLUE}= {Fore.YELLOW}${round(sum_cost, 3)}{Style.RESET_ALL}"
        )
        return sum_cost
