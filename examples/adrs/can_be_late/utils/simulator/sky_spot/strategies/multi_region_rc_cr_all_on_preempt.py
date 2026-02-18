"""
Multi-region RC-CR strategy that tries ALL regions when preempted.
"""

import argparse
import logging
import math
import typing

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType
from sky_spot.multi_region_types import TryLaunch, Terminate, Action, LaunchResult

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class MultiRegionRCCRAllOnPreemptStrategy(MultiRegionStrategy):
    """
    RC-CR strategy that tries all regions when preempted.
    When a SPOT instance gets preempted, it will attempt to launch
    SPOT in ALL regions before falling back to ON_DEMAND.
    """
    
    NAME = 'multi_region_rc_cr_all_on_preempt'
    
    def __init__(self, args):
        super().__init__(args)
        self.cur_on_demand_time = 0
        self.last_spot_region = None  # Track which region had SPOT
        
    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.cur_on_demand_time = 0
        self.last_spot_region = None
        
    def _condition(self):
        """Condition for switching from NONE to SPOT/ON_DEMAND."""
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0
    
    def _condition2(self):
        """Condition for switching from ON_DEMAND back to SPOT."""
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        c_t = self.task_duration - sum(self.task_done_time)
        c_0 = self.task_duration
        r_0 = self.deadline
        return (t + 2 * d) * c_0 / r_0 - (c_0 - c_t)
    
    def _was_preempted(self, active_instances: dict) -> bool:
        """Check if we were preempted (had SPOT before, don't have it now)."""
        # If we had a SPOT instance but now have no active instances, we were preempted
        if self.last_spot_region is not None and len(active_instances) == 0:
            return True
        # If we had SPOT but now only have ON_DEMAND, we were preempted
        if self.last_spot_region is not None and len(active_instances) > 0:
            cluster_types = list(active_instances.values())
            if ClusterType.SPOT not in cluster_types:
                return True
        return False
    
    def _step_multi(self) -> typing.Generator[Action, typing.Optional[LaunchResult], None]:
        """Multi-region strategy that tries all regions when preempted."""
        # Check if task is done
        remaining_task_seconds = self.task_duration - sum(self.task_done_time)
        if remaining_task_seconds <= 1e-3:
            active_instances = self.env.get_active_instances()
            for region in active_instances:
                yield Terminate(region=region)
            return
        
        # Get current state
        active_instances = self.env.get_active_instances()
        assert len(active_instances) <= 1, "Should maintain at most one instance"
        
        last_cluster_type = ClusterType.NONE
        last_region = None
        if active_instances:
            last_region = list(active_instances.keys())[0]
            last_cluster_type = active_instances[last_region]
        
        # Check if we were preempted
        was_preempted = self._was_preempted(active_instances)
        
        # Update tracking for next tick
        if last_cluster_type == ClusterType.SPOT:
            self.last_spot_region = last_region
        elif last_cluster_type == ClusterType.ON_DEMAND:
            # Don't clear last_spot_region yet, we might have been preempted
            pass
        
        # If preempted, terminate and try ALL regions
        if was_preempted:
            logger.info(f"[{self.NAME}] Detected preemption! Trying all {self.env.num_regions} regions...")
            
            # Terminate the preempted instance
            if last_region is not None:
                yield Terminate(region=last_region)
            
            # Try SPOT in ALL regions
            launched = False
            for region in range(self.env.num_regions):
                logger.debug(f"[{self.NAME}] Trying SPOT in region {region}...")
                result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                assert result is not None
                if result.success:
                    logger.info(f"[{self.NAME}] Successfully launched SPOT in region {region}")
                    launched = True
                    self.last_spot_region = None  # Clear preemption tracking
                    break
            
            if not launched:
                logger.info(f"[{self.NAME}] No SPOT available in any region, launching ON_DEMAND")
                # No SPOT available anywhere, use ON_DEMAND
                result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                assert result is not None
                assert result.success, "ON_DEMAND should always succeed"
            
            return
        
        # Normal RC-CR logic (not preempted)
        has_spot = True  # Optimistic assumption
        request_type = self._compute_request_type(last_cluster_type, has_spot)
        
        # Handle termination or no change
        if last_region is not None:
            if request_type == ClusterType.NONE or request_type != last_cluster_type:
                yield Terminate(region=last_region)
                if request_type == ClusterType.NONE:
                    return
            else:
                # Already have what we want
                return
        
        # Try to launch the requested instance type
        if request_type == ClusterType.SPOT:
            # Normal case: just try the first available region
            launched = False
            for region in range(self.env.num_regions):
                result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                assert result is not None
                if result.success:
                    launched = True
                    break
            
            if not launched:
                # No SPOT available, recompute with has_spot=False
                has_spot = False
                request_type = self._compute_request_type(ClusterType.NONE, has_spot)
                
                if request_type == ClusterType.ON_DEMAND:
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result is not None
                    assert result.success
        
        elif request_type == ClusterType.ON_DEMAND:
            result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
            assert result is not None
            assert result.success
    
    def _compute_request_type(self, current_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Compute the request type using RC-CR-Threshold logic."""
        remaining_time = math.floor(
            (self.deadline - self.env.elapsed_seconds) / self.env.gap_seconds
        ) * self.env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        
        # Default decision
        request_type = ClusterType.SPOT if has_spot else ClusterType.NONE
        
        # Apply conditions
        if self._condition() < 0:
            request_type = ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND
        
        if current_cluster_type == ClusterType.ON_DEMAND:
            if not has_spot or self._condition2() >= 0:
                request_type = ClusterType.ON_DEMAND
        
        # Check deadlines
        total_task_remaining = math.ceil(
            (remaining_task_time + self.restart_overhead) / self.env.gap_seconds
        ) * self.env.gap_seconds
        total_task_remaining_with_2D = math.ceil(
            (remaining_task_time + 2 * self.restart_overhead) / self.env.gap_seconds
        ) * self.env.gap_seconds
        
        if (total_task_remaining_with_2D >= remaining_time and 
            current_cluster_type in [ClusterType.NONE, ClusterType.ON_DEMAND]):
            request_type = ClusterType.ON_DEMAND
        
        if total_task_remaining >= remaining_time:
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.ON_DEMAND
            if self.restart_overhead == 0 and has_spot:
                request_type = ClusterType.SPOT
        
        # Update on_demand time tracking
        if request_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time += self.env.gap_seconds
        elif current_cluster_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time = 0
        
        return request_type
    
    @classmethod
    def _from_args(cls, parser: argparse.ArgumentParser) -> 'MultiRegionRCCRAllOnPreemptStrategy':
        return cls(parser.parse_args()) 