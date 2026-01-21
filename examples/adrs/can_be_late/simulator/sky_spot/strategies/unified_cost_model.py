"""
Pure Unified Cost Model Strategy

A truly pure implementation of the unified cost framework based on time value theory.
No heuristics, no thresholds, just pure cost-based decisions.
"""

import argparse
import logging
import typing

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot import env, task
from sky_spot.multi_region_types import (
    TryLaunch,
    Terminate,
    Action,
    LaunchResult,
    ClusterType
)
from sky_spot.utils import COST_K, COSTS

logger = logging.getLogger(__name__)

# Hardcoded region characteristics from offline analysis
REGION_CHARACTERISTICS = {
    0: {'avg_availability': 0.2193, 'avg_duration': 0.7282},  # us-east-1a
    1: {'avg_availability': 0.5385, 'avg_duration': 2.3524},  # us-east-1d
    2: {'avg_availability': 0.5587, 'avg_duration': 2.1241},  # us-east-1c
    3: {'avg_availability': 0.6186, 'avg_duration': 2.5427},  # us-east-1f
    4: {'avg_availability': 0.8657, 'avg_duration': 12.4883}, # us-east-2a
}

class UnifiedCostModelStrategy(MultiRegionStrategy):
    NAME = 'unified_cost_model'
    
    def __init__(self, args):
        super().__init__(args)
        
        # Cost parameters ($/hour) from sky_spot/utils.py
        self.c_spot = COSTS[ClusterType.SPOT]    # SPOT cost per hour
        self.c_od = COSTS[ClusterType.ON_DEMAND]       # ON_DEMAND cost per hour
        
        logger.info(f"Initialized Unified Cost Model strategy")
    
    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self._last_duration = 0
        logger.info("Unified Cost Model strategy reset")
    
    def _compute_time_value(self) -> float:
        """
        Compute the time value V(t,p) based on unified theory.
        
        V(t,p) = C_OD × (D-p)/(T-t) × T/D
        
        This represents the marginal value of progress per unit time.
        """
        D = self.task_duration  # Total task (seconds)
        T = self.deadline       # Deadline (seconds)
        p = sum(self.task_done_time)  # Progress (seconds)
        t = self.env.elapsed_seconds  # Elapsed time (seconds)
        
        remaining_task = D - p
        remaining_time = T - t
        
        assert remaining_time > 0
        
        if remaining_task <= 0:
            # Task done - no value
            return 0
        
        # Core formula: V = C_OD × (D-p)/(T-t) × T/D
        # Simplifies to: V = C_OD × T × (D-p) / (D × (T-t))
        V_per_second = self.c_od * (T / D) * (remaining_task / remaining_time) / 3600
        V_per_hour = V_per_second * 3600
        
        return V_per_hour
    
    def _compute_spot_net_value(self, region_id: int, V: float) -> float:
        """
        Compute expected net value of SPOT in a region.
        
        Net value = Expected progress value - Expected cost
        """
        duration = REGION_CHARACTERISTICS[region_id]['avg_duration']
        
        # Account for restart overhead
        effective_duration = max(0, duration - self.restart_overhead / 3600)
        
        if effective_duration <= 0:
            return -float('inf')
        
        # Expected net value over the duration
        # Value gained = V × effective_duration
        # Cost incurred = c_spot × duration
        value_gained = V * effective_duration
        cost_incurred = self.c_spot * duration
        net_value = value_gained - cost_incurred
        
        # Normalize to per-hour for comparison
        net_value_per_hour = net_value / duration

        logger.warning(f"For region {region_id}, net value per hour = {net_value_per_hour:.3f}")
        logger.warning(f"That's because net_value({net_value_per_hour}) = V({V}) * effective_duration({effective_duration}) - c_spot({self.c_spot}) * duration({duration})")
        
        return net_value_per_hour
    
    def _compute_od_net_value(self, V: float) -> float:
        """
        Compute net value of ON_DEMAND.
        
        For ON_DEMAND: net value = V - C_OD
        """
        return V - self.c_od
    
    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Pure unified cost model decision making - simplified version."""
        
        env = typing.cast('env.MultiTraceEnv', self.env)
        
        # Get current state
        active_instances = env.get_active_instances()
        
        # Check if task is done
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task <= 1e-3:
            # Task complete - terminate everything
            for region in list(active_instances.keys()):
                yield Terminate(region=region)
            return
        
        # Compute current time value
        V = self._compute_time_value()
        logger.warning(f"V={V:.3f}")
        
        od_value = self._compute_od_net_value(V)

        # Apply migration penalty if we have an active instance
        migration_penalty = 0
        current_region = None
        current_type: ClusterType = ClusterType.NONE
        current_value = 0 # Waiting
        if active_instances:
            assert len(active_instances) == 1, active_instances
            current_region = list(active_instances.keys())[0]
            current_type = active_instances[current_region]
            migration_penalty = self.restart_overhead / 3600 * V
            
            # Get current instance value for comparison
            if current_type == ClusterType.SPOT:
                current_value = self._compute_spot_net_value(current_region, V)
            elif current_type == ClusterType.ON_DEMAND:
                current_value = od_value
            else:
                assert Falsep
            
            logger.info(f"Current: {current_type.name} in R{current_region} (value={current_value:.3f})")
        
        # Build unified list of all options with their values
        # If we have active instances, stopping means we'll need to restart later
        wait_value = 0.0
        if active_instances:
            # Penalize waiting because we'll need to pay restart overhead later
            future_restart_cost = (self.restart_overhead / 3600) * V
            wait_value = -future_restart_cost * 0.5  # Discount factor for future cost
        
        all_options: list[tuple[float, str, typing.Optional[int]]] = [(wait_value, 'NONE', None)]
        
        # Add all SPOT options
        for region_id in range(env.num_regions):
            value = self._compute_spot_net_value(region_id, V)
            all_options.append((value - migration_penalty, 'SPOT', region_id))
        
        # Add ON_DEMAND option
        all_options.append((od_value - migration_penalty, 'ON_DEMAND', 0))

        # Sort by value (descending)
        all_options.sort(key=lambda x: x[0], reverse=True)

        # Drop all options with negative value
        all_options = [option for option in all_options if option[0] >= 0]
        
        # Try options in order until we find something better than current
        for value, cluster_type_str, region in all_options:
            assert value >= 0
            
            # Should be checked first, like current WAITING and still WAITING
            if value <= current_value:
                break

            # Skip current instance
            if region == current_region and cluster_type_str == current_type.name:
                continue


            launch_success = False

            # Try to launch this option
            if cluster_type_str == 'SPOT':
                assert region is not None
                logger.info(f"Trying SPOT in region {region} (value={value:.3f})")
                result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                assert result is not None
                launch_success = result.success
            elif cluster_type_str == 'ON_DEMAND':
                assert region is not None
                logger.info(f"Launching ON_DEMAND (value={value:.3f})")
                result = yield TryLaunch(region=region, cluster_type=ClusterType.ON_DEMAND)
                assert result is not None
                assert result.success  # ON_DEMAND always succeeds
                launch_success = True
            else:
                # We should wait, and terminate current instance
                launch_success = True

            if launch_success:
                if current_region is not None:
                    logger.info(f"Terminating old instance {current_type.name} in R{current_region}")
                    yield Terminate(region=current_region)
                return
        
        logger.info(f"No viable options, keeping current {current_type.name} in R{current_region}")

    @classmethod
    def _from_args(cls, parser):
        return cls(parser.parse_args())