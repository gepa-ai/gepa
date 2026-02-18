"""
Reactive Multi-region RC/CR Strategy.
Only considers other regions when the current region's spot is preempted.
Otherwise, stays in the same region (sticky behavior).
"""

import argparse
import logging
import typing

from sky_spot.strategies.multi_region_rc_cr_threshold import MultiRegionRCCRThresholdStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot.multi_region_types import Action, LaunchResult
    from sky_spot import task

logger = logging.getLogger(__name__)


class MultiRegionRCCRReactiveStrategy(MultiRegionRCCRThresholdStrategy):
    """Reactive multi-region strategy - only switches regions when preempted."""
    NAME = 'multi_region_rc_cr_reactive'
    
    def __init__(self, args):
        super().__init__(args)
        self.preferred_region = None  # Track which region we're "sticking" to
    
    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.preferred_region = None
    
    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Reactive version - only considers other regions when preempted."""
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
            # If we have an active instance, that's our preferred region
            self.preferred_region = last_region
        
        # Check if task is done
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            if last_region is not None:
                yield Terminate(region=last_region)
            return
        
        # Check if we were preempted (had SPOT but now it's gone)
        was_preempted = False
        if last_region is None and self.preferred_region is not None:
            # We had a preferred region but no active instance - likely preempted
            was_preempted = True
        
        # Compute decision using single-region logic
        # For reactive strategy, we assume spot is available and let TryLaunch check
        has_spot = True
        request_type = self._compute_request_type(last_cluster_type, has_spot)
        
        # Handle termination or no change
        if last_region is not None:
            if request_type == ClusterType.NONE or request_type != last_cluster_type:
                yield Terminate(region=last_region)
                if request_type == ClusterType.NONE:
                    return
            else:
                # Already have what we want in current region
                return
        
        # Try to launch the requested instance type
        if request_type == ClusterType.SPOT:
            launched = False
            
            # First, try preferred region if we have one
            if self.preferred_region is not None:
                result = yield TryLaunch(region=self.preferred_region, cluster_type=ClusterType.SPOT)
                assert result is not None, "TryLaunch should always return a result"
                if result.success:
                    launched = True
                    logger.debug(f"Launched SPOT in preferred region {self.preferred_region}")
            
            # Only try other regions if we were preempted or don't have a preferred region
            if not launched and (was_preempted or self.preferred_region is None):
                logger.debug(f"Reactive strategy: Trying other regions (was_preempted={was_preempted})")
                for region in range(env.num_regions):
                    if region == self.preferred_region:
                        continue  # Already tried
                    result = yield TryLaunch(region=region, cluster_type=ClusterType.SPOT)
                    assert result is not None, "TryLaunch should always return a result"
                    if result.success:
                        launched = True
                        self.preferred_region = region  # Update preferred region
                        logger.debug(f"Launched SPOT in new region {region}")
                        break
            
            if not launched:
                # No SPOT available anywhere, fall back to ON_DEMAND
                has_spot = False
                request_type = self._compute_request_type(ClusterType.NONE, has_spot)
                
                if request_type == ClusterType.ON_DEMAND:
                    # Launch ON_DEMAND in preferred region if possible
                    launch_region = self.preferred_region if self.preferred_region is not None else 0
                    result = yield TryLaunch(region=launch_region, cluster_type=ClusterType.ON_DEMAND)
                    assert result is not None, "TryLaunch should always return a result"
                    assert result.success, "ON_DEMAND should always succeed"
                    if self.preferred_region is None:
                        self.preferred_region = launch_region
                # else: request_type is NONE, don't launch anything
        
        elif request_type == ClusterType.ON_DEMAND:
            # Launch ON_DEMAND in preferred region if possible
            launch_region = self.preferred_region if self.preferred_region is not None else 0
            result = yield TryLaunch(region=launch_region, cluster_type=ClusterType.ON_DEMAND)
            assert result is not None, "TryLaunch should always return a result"
            assert result.success, "ON_DEMAND should always succeed"
            if self.preferred_region is None:
                self.preferred_region = launch_region
    
    @classmethod
    def _from_args(cls, parser: 'argparse.ArgumentParser') -> 'MultiRegionRCCRReactiveStrategy':
        # Just use parent's argument parsing
        args, _ = parser.parse_known_args()
        return cls(args) 