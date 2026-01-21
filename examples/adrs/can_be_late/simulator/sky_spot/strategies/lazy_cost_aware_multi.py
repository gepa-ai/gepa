"""Lazy cost-aware strategy for multi-region scheduling (adapted from OpenEvolve's best_program.py)"""

import configargparse
import json
import logging
import typing
from typing import Optional, List

from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env as env_lib
    from sky_spot import task
    from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action

logger = logging.getLogger(__name__)


class Stats:
    """Track throughput statistics with momentum"""
    def __init__(self, momentum: float = 0.5):
        self.momentum = momentum
        self.current = None
    
    def update(self, total_work_done: float, was_computing: bool):
        """Update throughput based on current progress"""
        if not was_computing:
            return
        if self.current is None:
            self.current = 1.0  # Initial guess
        else:
            # Exponential moving average
            self.current = self.momentum * self.current + (1 - self.momentum) * 1.0
    
    def mean_throughput(self) -> float:
        """Get average throughput (normalized to 1.0 for perfect speed)"""
        return self.current if self.current is not None else 1.0


class RegionState:
    """Track state for multi-region exploration"""
    def __init__(self, num_regions: int):
        self.num_regions = num_regions
        self.last_spot_region: Optional[int] = None
        self.last_explored = 0
        self.region_failures: List[float] = [0.0] * num_regions
        self.cooldown_end: List[float] = [0.0] * num_regions
        
    def mark_spot_success(self, region: int):
        """Mark a region as having successful SPOT availability"""
        self.last_spot_region = region
        self.region_failures[region] = 0.0
        
    def mark_spot_failure(self, region: int, current_time: float):
        """Mark a region as having failed SPOT attempt"""
        self.region_failures[region] += 1
        # Exponential backoff: 2^failures minutes (capped at 30 min)
        cooldown_minutes = min(30, 2 ** self.region_failures[region])
        self.cooldown_end[region] = current_time + cooldown_minutes * 60
        
    def is_in_cooldown(self, region: int, current_time: float) -> bool:
        """Check if region is still in cooldown period"""
        return current_time < self.cooldown_end[region]
        
    def get_next_region_to_explore(self, current_time: float) -> Optional[int]:
        """Get next region to explore (round-robin, skipping cooldowns)"""
        for i in range(self.num_regions):
            region = (self.last_explored + i) % self.num_regions
            if not self.is_in_cooldown(region, current_time):
                self.last_explored = region
                return region
        return None  # All regions in cooldown


class LazyCostAwareMultiStrategy(MultiRegionStrategy):
    """
    Lazy cost-aware strategy adapted from OpenEvolve's best_program.py.
    
    Key features:
    - Conservative/lazy: prefers SPOT when possible
    - Only uses ON_DEMAND when no slack available
    - Implements intelligent region exploration with cooldown
    - Based on the successful lazy_cost_aware_v3 strategy
    """
    NAME = 'lazy_cost_aware_multi'
    
    def __init__(self, args: configargparse.Namespace):
        super().__init__(args)
        self._stats: Optional[Stats] = None
        self._regions: Optional[RegionState] = None
        self._current_region: Optional[int] = None
        self._was_computing_last_tick = False
    
    def reset(self, env: 'env_lib.Env', task: 'task.Task'):
        """Reset strategy state for new simulation"""
        super().reset(env, task)
        # Initialize stats and region tracking
        self._stats = Stats(momentum=0.5)
        
        # Get number of regions from environment
        if hasattr(env, 'trace_files'):
            num_regions = len(env.trace_files)
        else:
            num_regions = 2  # Default fallback
            
        self._regions = RegionState(num_regions)
        self._current_region = None
        self._was_computing_last_tick = False
        
        logger.info(f"[{self.NAME}] Initialized with {num_regions} regions")
    
    def _progress(self) -> float:
        """Get current progress fraction (0 to 1)"""
        return sum(self.task_done_time) / self.task_duration
        
    def _slack_seconds(self) -> float:
        """Calculate available slack time in seconds"""
        elapsed = self.env.elapsed_seconds
        remaining_wall_time = self.deadline - elapsed
        remaining_work = self.task_duration - sum(self.task_done_time)
        
        # Account for throughput based on stats
        throughput = self._stats.mean_throughput() if self._stats else 1.0
        estimated_compute_time = remaining_work / throughput
        
        return remaining_wall_time - estimated_compute_time
    
    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Main decision logic for multi-region scheduling"""
        from sky_spot.multi_region_types import TryLaunch, Terminate
        
        # Ensure we're initialized
        if self._stats is None:
            self.reset(self.env, self.task)
            
        # Cast to get proper type hints
        env = typing.cast('env_lib.MultiTraceEnv', self.env)
        
        # Get current state
        active_instances = env.get_active_instances()
        progress_now = self._progress()
        
        # Update statistics
        self._stats.update(progress_now, self._was_computing_last_tick)
        self._was_computing_last_tick = False
            
        # 1) If we already have a SPOT instance, just keep it
        if active_instances:
            # Get the first active region from the dictionary
            self._current_region = next(iter(active_instances.keys()))
            self._was_computing_last_tick = True
            return
            
        # 2) No instances running - check if we need ON_DEMAND
        seconds_left = self.deadline - self.env.elapsed_seconds
        need_seconds = self.task_duration - sum(self.task_done_time)
            
        # Only when we *strictly* need more compute-seconds than wall time
        if need_seconds > seconds_left >= 0:
            logger.warning(
            "[%s] No slack (need %.0f s of %.0f s) – starting ON_DEMAND",
            self.NAME, need_seconds, seconds_left
            )
            # Launch ON_DEMAND in any available region
            for region in range(self._regions.num_regions):
                result = yield TryLaunch(region=region, 
                                       cluster_type=ClusterType.ON_DEMAND)
                assert result is not None
                if result.success:
                    self._current_region = region
                    self._was_computing_last_tick = True
                    return
        
        # 3) We *do* have slack – stay cheap and explore
        slack_ratio = 1.0 - (need_seconds / seconds_left if seconds_left > 0 else 1.0)
        now = self.env.elapsed_seconds
        
        # 3a) Prefer the most recently known SPOT region
        if (self._regions.last_spot_region is not None
            and not self._regions.is_in_cooldown(self._regions.last_spot_region, now)):
            result = yield TryLaunch(region=self._regions.last_spot_region,
                                   cluster_type=ClusterType.SPOT)
            assert result is not None
            if result.success:
                self._current_region = self._regions.last_spot_region
                self._was_computing_last_tick = True
                self._regions.mark_spot_success(self._regions.last_spot_region)
                return
            else:
                # Mark failure and continue to exploration
                self._regions.mark_spot_failure(self._regions.last_spot_region, now)
        
        # 3b) Exploration: round-robin through regions (with cooldown)
        region_to_try = self._regions.get_next_region_to_explore(now)
        if region_to_try is not None:
            result = yield TryLaunch(region=region_to_try,
                                           cluster_type=ClusterType.SPOT)
            assert result is not None
            if result.success:
                self._current_region = region_to_try
                self._was_computing_last_tick = True
                self._regions.mark_spot_success(region_to_try)
                logger.info(f"[{self.NAME}] Found SPOT in region {region_to_try}")
            else:
                self._regions.mark_spot_failure(region_to_try, now)
                            
        # If no regions available or all failed, we simply wait this tick
        # The strategy is "lazy" - it prefers to wait rather than pay for ON_DEMAND
    
    @classmethod
    def _from_args(cls, parser):
        """Create strategy instance from command line arguments"""
        return cls(parser.parse_args()) 