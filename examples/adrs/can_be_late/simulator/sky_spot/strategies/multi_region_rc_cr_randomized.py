"""
Multi-region RC/CR with randomized region selection.
This randomly shuffles regions when looking for spot availability.
"""

import logging
import random
import typing

from sky_spot.strategies.multi_region_rc_cr_threshold import MultiRegionRCCRThresholdStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot.multi_region_types import Action, LaunchResult

logger = logging.getLogger(__name__)


class MultiRegionRCCRRandomizedStrategy(MultiRegionRCCRThresholdStrategy):
    NAME = 'multi_region_rc_cr_randomized'
    
    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Multi-region version with randomized region selection."""
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
            # Randomize region order when trying SPOT
            regions = list(range(env.num_regions))
            random.shuffle(regions)
            
            launched = False
            for region in regions:
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
