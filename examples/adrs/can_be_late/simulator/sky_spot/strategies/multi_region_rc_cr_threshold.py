"""
Our policy: Uniform Progress.
Multi-region version - finds ANY region with spot availability.
"""

import argparse
import json
import logging
import math

from sky_spot.strategies import rc_threshold
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

import typing
if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task
    from sky_spot.multi_region_types import Action, LaunchResult

logger = logging.getLogger(__name__)


class MultiRegionRCCRThresholdStrategy(rc_threshold.RCThresholdStrategy, MultiRegionStrategy):
    NAME = 'multi_region_rc_cr_threshold'

    def __init__(self, args):
        args.keep_on_demand = None
        super().__init__(args)

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.cur_on_demand_time = 0

    def _condition(self):
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0

    def _condition2(self):
        # When on VM, only switch back to spot only when the condition2 < 0
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        c_t = self.task_duration - sum(self.task_done_time)
        c_0 = self.task_duration
        r_0 = self.deadline
        return (t + 2 * d) * c_0 / r_0 - (c_0 - c_t)

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        # Not used for multi-region
        raise NotImplementedError("Multi-region strategy should use _step_multi")

    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Multi-region version - finds ANY region with spot availability."""
        from sky_spot.multi_region_types import TryLaunch, Terminate
        
        env = typing.cast('env.MultiTraceEnv', self.env)
        
        # Assert we only maintain one instance at a time
        active_instances = env.get_active_instances()
        assert len(active_instances) <= 1, f"Multi-region RC/CR should only have at most 1 instance, got {len(active_instances)}"
        
        # Get current state
        last_cluster_type = ClusterType.NONE
        last_region = None
        if active_instances:
            last_region = list(active_instances.keys())[0]
            last_cluster_type = active_instances[last_region]
        
        # Check if task is done
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            if last_region is not None:
                yield Terminate(region=last_region)
            return
        
        # Compute decision using single-region logic (assuming spot is available)
        has_spot = True
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
            # Try all regions for SPOT
            launched = False
            for region in range(env.num_regions):
                result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                assert result is not None, "TryLaunch should always return a result"
                if result.success:
                    launched = True
                    break
            
            if not launched:
                # No SPOT available, recompute with has_spot=False
                has_spot = False
                request_type = self._compute_request_type(ClusterType.NONE, has_spot)
                
                if request_type == ClusterType.ON_DEMAND:
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result is not None, "TryLaunch should always return a result"
                    assert result.success, "ON_DEMAND should always succeed"
                # else: request_type is NONE, don't launch anything
        
        elif request_type == ClusterType.ON_DEMAND:
            result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
            assert result is not None, "TryLaunch should always return a result"
            assert result.success, "ON_DEMAND should always succeed"

    def _compute_request_type(self, current_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Compute the request type using single-region logic."""
        env = self.env
        remaining_time = math.floor((self.deadline - env.elapsed_seconds) / env.gap_seconds) * env.gap_seconds
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
        total_task_remaining = math.ceil((remaining_task_time + self.restart_overhead) / env.gap_seconds) * env.gap_seconds
        total_task_remaining_with_2D = math.ceil((remaining_task_time + 2 * self.restart_overhead) / env.gap_seconds) * env.gap_seconds
        
        if total_task_remaining_with_2D >= remaining_time and current_cluster_type in [ClusterType.NONE, ClusterType.ON_DEMAND]:
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
            self.cur_on_demand_time += env.gap_seconds
        elif current_cluster_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time = 0
        
        return request_type

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser') -> 'MultiRegionRCCRThresholdStrategy':
        group = parser.add_argument_group('RCV2DThresholdStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
