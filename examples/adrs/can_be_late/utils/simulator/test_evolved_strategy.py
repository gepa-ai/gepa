#!/usr/bin/env python3
"""Direct test of evolved strategy."""

import sys
import os
sys.path.append('/home/andyl/cant-be-late')

from sky_spot.strategies.strategy import Strategy
from sky_spot.utils import ClusterType
from sky_spot.task import SingleTask
from sky_spot.env import TraceEnvironment
import json

# Create simple test strategy
class TestStrategy(Strategy):
    NAME = 'test_simple'
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task <= 1e-3:
            return ClusterType.NONE
        
        remaining_time = self.deadline - self.env.elapsed_seconds
        if remaining_task + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND
        
        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE
    
    @classmethod
    def _from_args(cls, parser):
        class Args:
            def __init__(self):
                self.deadline_hours = 52.0
                self.restart_overhead_hours = [0.2]
                self.task_duration_hours = [48.0]
        return cls(Args())

# Load a trace
trace_file = "data/real/ping_based/random_start_time/us-west-2a_k80_1/0.json"
with open(trace_file, 'r') as f:
    trace_data = json.load(f)

# Create environment
env = TraceEnvironment(trace_data['data'], trace_data['metadata']['gap_seconds'])

# Create task
task = SingleTask(48.0 * 3600)  # 48 hours in seconds

# Create strategy
class Args:
    def __init__(self):
        self.deadline_hours = 52.0
        self.restart_overhead_hours = [0.2]
        self.task_duration_hours = [48.0]

strategy = TestStrategy(Args())
strategy.reset(env, task)

# Run simulation
total_cost = 0.0
tick = 0
max_ticks = 1000

while tick < max_ticks:
    has_spot = env._spot_available()
    last_type = env.cluster_type
    
    decision = strategy.step(has_spot)
    
    if decision == ClusterType.NONE and task.is_done():
        print(f"Task completed at tick {tick}")
        break
    
    env.step(decision, strategy, task)
    tick += 1
    
    # Track cost
    if env.cluster_type == ClusterType.SPOT:
        total_cost += env.spot_cost * env.gap_seconds / 3600
    elif env.cluster_type == ClusterType.ON_DEMAND:
        total_cost += env.on_demand_cost * env.gap_seconds / 3600

print(f"Total cost: ${total_cost:.2f}")
print(f"Task done: {task.is_done()}")
print(f"Deadline met: {env.elapsed_seconds <= 52.0 * 3600}")