import math
import json
import logging
import math
import os
import typing
from typing import Dict, List, Tuple, Type, Sequence, Optional, Any
import abc

from sky_spot import trace
from sky_spot.utils import ClusterType, COSTS, DEVICE_COSTS, COST_K
from sky_spot import task as task_lib

if typing.TYPE_CHECKING:
    import configargparse
    from sky_spot.strategies.strategy import MultiRegionStrategy

logger = logging.getLogger(__name__)


class Env(abc.ABC):
    NAME = 'abstract'
    SUBCLASSES: Dict[str, Type['Env']] = {}

    def __init__(self, gap_seconds: float):
        self.gap_seconds = gap_seconds
        self.reset()

    def reset(self):
        # dones not include the cluster_type for the current timestamp - 1 -> timestamp, until observed on timestamp
        self.cluster_type_histroy = []
        self.cluster_type = ClusterType.NONE
        self.tick = 0
        self.observed_tick = -1

    def __init_subclass__(cls) -> None:
        assert cls.NAME not in cls.SUBCLASSES and cls.NAME != 'abstract', f'Name {cls.NAME} already exists'
        cls.SUBCLASSES[cls.NAME] = cls

    def spot_available(self) -> bool:
        """
        Returns True if spot is available at the current timestamp -> timestamp + 1
        """
        raise NotImplementedError

    def observe(self) -> Tuple[ClusterType, bool]:
        """
        Returns the cluster type (at last time gap) and whether spot is available
        """
        assert self.observed_tick == self.tick - 1, (self.observed_tick,
                                                     self.tick)
        self.observed_tick = self.tick
        has_spot = self.spot_available()
        last_cluster_type = self.cluster_type
        self.cluster_type_histroy.append(last_cluster_type)

        if self.cluster_type == ClusterType.SPOT and not has_spot:
            logger.debug(f'Preempted at {self.tick}')
            self.cluster_type = ClusterType.NONE
        return last_cluster_type, has_spot

    def step(self, request_type: ClusterType):
        if self.observed_tick != self.tick:
            self.observe()
        if request_type == ClusterType.SPOT and not self.spot_available():
            raise ValueError('Spot not available')
        new_cluster_type = self._step(request_type)
        self.tick += 1
        return new_cluster_type

    def _step(self, request_type: ClusterType):
        self.cluster_type = request_type
        return self.cluster_type

    def get_trace_before_end(self, end: float) -> trace.Trace:
        # Used for ideal strategy
        raise NotImplementedError

    @property
    def elapsed_seconds(self) -> float:
        return self.tick * self.gap_seconds

    @property
    def accumulated_cost(self) -> float:
        """Accumulated cost of the environment"""
        costs_map = self.get_constant_cost_map()
        return sum(costs_map[cluster_type] * self.gap_seconds / 3600
                   for cluster_type in self.cluster_type_histroy)

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        return COSTS

    def info(self) -> dict:
        # Step should have been called
        assert self.tick == self.observed_tick + 1
        return {
            'Timestamp':
            self.tick - 1,
            'Elapsed': (self.tick - 1) * self.gap_seconds,
            'Cost':
            self.accumulated_cost,
            'ClusterType':
            self.cluster_type_histroy[-1].value
            if self.cluster_type_histroy else ClusterType.NONE.value,
        }

    def __repr__(self) -> str:
        return f'{self.NAME}({json.dumps(self.config)})'

    @property
    def config(self):
        return dict()

    @classmethod
    def from_args(cls,
                  parser: 'configargparse.ArgumentParser') -> Sequence['Env']:
        # parser.add_argument(f'--env-config', type=str, default=None, is_config_file=True, required=False)
        parser.add_argument('--env',
                            type=str,
                            default='trace',
                            choices=cls.SUBCLASSES.keys())
        args, _ = parser.parse_known_args()
        cls = cls.SUBCLASSES[args.env]
        return cls._from_args(parser)

    @classmethod
    def _from_args(cls,
                   parser: 'configargparse.ArgumentParser') -> Sequence['Env']:
        raise NotImplementedError


class TraceEnv(Env):
    NAME = 'trace'

    def __init__(self, trace_file: str, env_start_hours: float):

        self._trace_file = trace_file
        self.trace: trace.Trace = trace.Trace.from_file(trace_file)

        self._start_index = 0
        if env_start_hours > 0:
            self._start_index = int(
                math.ceil(env_start_hours * 3600 / self.trace.gap_seconds))

        for device, cost in DEVICE_COSTS.items():
            if device in trace_file:
                self._base_price = cost
                break
        assert self._base_price is not None, f'No base price found for {trace_file}'

        self._spot_price = None
        if self.trace.get_price(0) is None:
            self._spot_price = self._base_price / COST_K

        super().__init__(self.trace.gap_seconds)

    def spot_available(self) -> bool:
        tick = self.tick + self._start_index
        if tick >= len(self.trace):
            raise ValueError(
                f'Timestamp {tick} out of range {len(self.trace)}')
        return not self.trace[tick]

    def get_trace_before_end(self, end: float) -> trace.Trace:
        end_index = int(math.ceil(end / self.gap_seconds))
        return self.trace[self._start_index:end_index + self._start_index] # type: ignore

    def next_wait_spot_length(self) -> Tuple[int, int]:
        wait_length = 0
        spot_length = 0
        start = self.tick + self._start_index
        if not self.spot_available():
            for i in range(start, len(self.trace)):
                if not self.trace[i]:
                    start = i
                    break
                wait_length += 1

        for i in range(start, len(self.trace)):
            if not self.trace[i]:
                spot_length += 1
            else:
                break
        return wait_length, spot_length

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        return {
            ClusterType.ON_DEMAND:
            float(self._base_price),
            ClusterType.SPOT:
            float(self.trace.get_price(0)) # type: ignore
            if self._spot_price is None else float(self._spot_price),
            ClusterType.NONE:
            0.0,
        }

    def get_price(self) -> Dict[ClusterType, float]:
        if self._spot_price is not None:
            return {
                ClusterType.ON_DEMAND: float(self._base_price),
                ClusterType.SPOT: float(self._spot_price),
                ClusterType.NONE: 0.0,
            }
        spot_price = self.trace.get_price(self.tick + self._start_index)
        assert spot_price is not None, 'Spot price not available'
        return {
            ClusterType.ON_DEMAND: float(self._base_price),
            ClusterType.SPOT: spot_price,
            ClusterType.NONE: 0.0,
        }

    @property
    def config(self) -> dict:
        return {
            'name': self.NAME,
            'trace_file': self._trace_file,
            'start_index': self._start_index,
            'metadata': self.trace.metadata,
            'tace_file': self._trace_file
        }

    @classmethod
    def _from_args(
            cls,
            parser: 'configargparse.ArgumentParser') -> Sequence['TraceEnv']:
        group = parser.add_argument_group('TraceEnv')
        group.add_argument('--trace-file',
                           type=str,
                           help='File/folder containing the trace')
        group.add_argument('--env-start-hours',
                           type=float,
                           default=0,
                           help='Start hours of the trace')
        args, _ = parser.parse_known_args()
        return cls.create_env(args.trace_file, args.env_start_hours)

    @classmethod
    def create_env(cls, trace_file_or_dir: str,
                   env_start_hours: float) -> Sequence['TraceEnv']:
        if os.path.isdir(trace_file_or_dir):
            trace_files = []
            for file in sorted(os.listdir(trace_file_or_dir),
                               key=lambda x: int(x.split('.')[0])):
                # logger.debug(file)
                if file.endswith('.json'):
                    trace_files.append(os.path.join(trace_file_or_dir, file))
            return [
                cls(trace_file, env_start_hours) for trace_file in trace_files
            ]
        return [cls(trace_file_or_dir, env_start_hours)]


class MultiTraceEnv(Env):
    NAME = 'multi_trace'

    def __init__(self, trace_files: List[str], env_start_hours: float):
        self._trace_files = trace_files
        self.envs = [
            TraceEnv(trace_file, env_start_hours) for trace_file in trace_files
        ]
        self.num_regions = len(trace_files)

        gap_seconds = self.envs[0].trace.gap_seconds
        for env in self.envs:
            assert env.trace.gap_seconds == gap_seconds, "All traces must have the same gap seconds"

        # Multi-region specific state
        self.active_instances: Dict[int, ClusterType] = {}
        self.cost_history: List[Dict[int, ClusterType]] = []
        self.current_tick_costs: Dict[int, ClusterType] = {}
        
        # Simple flag to track new launches
        self._new_launch_this_tick = False
        self._had_new_launch_last_tick = False
        
        # Track region switches for dynamic migration time
        self._last_active_region: Optional[int] = None
        self._region_switch_info: Optional[Dict[str, Any]] = None
        
        # Track total number of migrations
        self.migration_count: int = 0

        super().__init__(gap_seconds)
        logger.debug(
            f"MultiTraceEnv initialized with {self.num_regions} regions: {trace_files}"
        )

    def reset(self):
        super().reset()
        for env in self.envs:
            env.reset()
        self.active_instances = {}
        self.cost_history = []
        self.current_tick_costs = {}
        self._new_launch_this_tick = False
        self._had_new_launch_last_tick = False
        self._last_active_region = None
        self._region_switch_info = None
        self.migration_count = 0
        self._launched_this_tick = {}
        self._pending_overwrites = {}
        logger.debug("MultiTraceEnv reset completed")

    def spot_available(self) -> bool:
        # MultiTraceEnv doesn't have a single spot availability
        raise NotImplementedError("Use _spot_available_in_region() for multi-region environment")

    def _spot_available_in_region(self, region_idx: int) -> bool:
        """Check spot availability in a specific region (internal use only)."""
        assert 0 <= region_idx < self.num_regions, f'Region index {region_idx} out of range'
        # Check if tick is within the valid range for this region
        sub_env = self.envs[region_idx]
        max_tick = len(sub_env.trace) - sub_env._start_index - 1
        if self.tick > max_tick:
            # If tick is beyond this region's trace, raise error
            raise ValueError(
                f'Simulation exceeded trace data bounds: tick {self.tick} > max tick {max_tick} '
                f'(region {region_idx}). This could mean: '
                f'1) Task took longer than expected to complete, possibly missing deadline; '
                f'2) Trace file is shorter than the deadline duration ({len(sub_env.trace)} ticks); '
                f'3) Strategy is not terminating properly. '
                f'Check task progress and deadline settings.'
            )
        # CRITICAL: Sync the tick of the region with the switcher's tick
        sub_env.tick = self.tick
        return sub_env.spot_available()

    def get_all_regions_spot_prices(self) -> List[Optional[float]]:
        """Return spot prices for all regions. If a region's spot is not available, its price is None."""
        prices = []
        for i in range(self.num_regions):
            sub_env = self.envs[i]
            max_tick = len(sub_env.trace) - sub_env._start_index - 1
            if self.tick > max_tick:
                # If tick is beyond this region's trace, raise error
                raise ValueError(
                    f'Simulation exceeded trace data bounds: tick {self.tick} > max tick {max_tick} '
                    f'(region {i}). This could mean: '
                    f'1) Task took longer than expected to complete, possibly missing deadline; '
                    f'2) Trace file is shorter than the deadline duration ({len(sub_env.trace)} ticks); '
                    f'3) Strategy is not terminating properly. '
                    f'Check task progress and deadline settings.'
                )
            # Sync tick before getting any info from sub-env
            sub_env.tick = self.tick
            if sub_env.spot_available():
                price_map = sub_env.get_price()
                prices.append(price_map.get(ClusterType.SPOT))
            else:
                prices.append(None)
        return prices

    def observe(self) -> Tuple[ClusterType, bool]:
        assert self.observed_tick == self.tick - 1, (self.observed_tick,
                                                     self.tick)
        self.observed_tick = self.tick
        
        # Finalize costs from previous tick
        if self.tick > 0:
            self._finalize_tick_costs()
        
        # Update launch flags after finalizing costs
        # This ensures the flag reflects launches from the previous tick
        self._had_new_launch_last_tick = self._new_launch_this_tick
        self._new_launch_this_tick = False
        
        # Clear tracking for new tick
        self._launched_this_tick = {}
        self._pending_overwrites = {}
        
        # Handle preemptions in all regions
        for region in list(self.active_instances.keys()):
            if self.active_instances[region] == ClusterType.SPOT:
                if not self._spot_available_in_region(region):
                    logger.debug(
                        f'Preempted at tick {self.tick} in region {region}'
                    )
                    self.active_instances.pop(region)
        
        # Multi-region env doesn't use this interface anymore
        # This method is only here because it's abstract in base class
        return ClusterType.NONE, False

    def get_trace_before_end(self, end: float) -> trace.Trace:
        # Multi-region env doesn't have a single trace
        raise NotImplementedError("Multi-region environment doesn't have a single trace")

    def get_num_regions(self) -> int:
        """Return the number of available regions."""
        return self.num_regions
    
    def get_active_instances(self) -> Dict[int, ClusterType]:
        """Get currently active instances across all regions."""
        return self.active_instances.copy()
    
    def get_region_name(self, region_idx: int) -> str:
        """Get the region name from trace filename."""
        trace_file = self._trace_files[region_idx]
        # Extract region name from path like 'data/.../us-east-1a_v100_1/0.json'
        import os
        region_dir = os.path.basename(os.path.dirname(trace_file))
        return region_dir

    def execute_multi_strategy(self, strategy: 'MultiRegionStrategy'):
        """Execute one step of a multi-region strategy using yield/generator pattern."""
        from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult
        
        # Reset safety net flag at the beginning of each tick
        self._safety_net_active = False
        
        # SAFETY NET: Apply strong guarantee before executing strategy
        # This ensures all multi-region strategies respect deadline constraints
        remaining_time_seconds = math.floor(
            (strategy.deadline - self.elapsed_seconds) / self.gap_seconds
        ) * self.gap_seconds
        remaining_task_seconds = strategy.task_duration - sum(strategy.task_done_time)
        
        # Check if we need to apply strong guarantee
        if remaining_task_seconds > 1e-3:  # Task not done yet
            total_task_remaining = math.ceil(
                (remaining_task_seconds + strategy.restart_overhead) / self.gap_seconds
            ) * self.gap_seconds
            
            if total_task_remaining >= remaining_time_seconds:
                # We're in a critical situation - need to ensure ON_DEMAND
                active_instances = self.get_active_instances()
                
                # Check if we have a working SPOT instance with no restart overhead
                has_working_spot = False
                for region, cluster_type in active_instances.items():
                    if cluster_type == ClusterType.SPOT and strategy.remaining_restart_overhead < 1e-3:
                        has_working_spot = True
                        break
                
                if not has_working_spot:
                    # Need ON_DEMAND to meet deadline
                    logger.warning(
                        "[SAFETY NET] Strong guarantee override: forcing ON_DEMAND "
                        "(remaining time: %.0f s, needed: %.0f s)",
                        remaining_time_seconds, total_task_remaining
                    )
                    
                    # First terminate any non-ON_DEMAND instances
                    for region, cluster_type in active_instances.items():
                        if cluster_type != ClusterType.ON_DEMAND:
                            self._terminate_internal(region)
                    
                    # Then launch ON_DEMAND if we don't have one
                    if ClusterType.ON_DEMAND not in active_instances.values():
                        # Try to launch ON_DEMAND in any region
                        return_code = self._try_launch_internal(0, ClusterType.ON_DEMAND)
                        assert return_code, "Should have launched ON_DEMAND in region 0"
                    
                    # Safety net has taken over - ignore all strategy actions this tick
                    self._safety_net_active = True
                    return
        
        # Normal strategy execution
        # Don't initialize costs here - let actions and finalize handle it
        # This allows terminating at tick start without being charged
        
        # Track instances launched in this tick to prevent same-tick termination of the same instance
        # Map region -> cluster_type that was launched this tick
        launched_this_tick = {}
        
        # Get the generator from strategy
        gen = strategy._step_multi()
        
        try:
            action = next(gen)
            while True:
                result = None
                
                if isinstance(action, TryLaunch):
                    # Execute launch attempt
                    success = self._try_launch_internal(action.region, action.cluster_type)
                    if success:
                        launched_this_tick[action.region] = action.cluster_type
                    result = LaunchResult(
                        success=success, 
                        region=action.region,
                        cluster_type=action.cluster_type if success else None
                    )
                    
                elif isinstance(action, Terminate):
                    # If safety net is active, ignore terminate actions
                    if hasattr(self, '_safety_net_active') and self._safety_net_active:
                        logger.warning(
                            f"Ignoring terminate action for region {action.region} - safety net is active"
                        )
                        result = None
                    else:
                        # Check if trying to terminate the SAME instance type that was just launched
                        # (allow terminating a different type to switch instance types)
                        if action.region in launched_this_tick:
                            launched_type = launched_this_tick[action.region]
                            current_type = self.active_instances.get(action.region)
                            if launched_type == current_type:
                                raise ValueError(
                                    f"Cannot terminate {launched_type.name} instance in region {action.region} "
                                    f"in the same tick it was launched. Minimum billing unit is one tick."
                                )
                        # Execute termination
                        self._terminate_internal(action.region)
                        result = None
                else:
                    raise ValueError(f"Unknown action type: {type(action)}")
                
                # Send result back and get next action
                action = gen.send(result)
                
        except StopIteration:
            # Strategy finished for this tick
            # Validate that no region has multiple instances
            self._validate_no_multiple_instances()
    
    def _try_launch_internal(self, region: int, cluster_type: ClusterType) -> bool:
        """Internal method to try launching an instance in a region."""
        # Track all launch attempts for validation later
        if not hasattr(self, '_launched_this_tick'):
            self._launched_this_tick = {}
        
        # Record this launch attempt
        if region in self._launched_this_tick:
            # Multiple launches in same region - track them all for validation
            logger.debug(f"Multiple launch attempts in region {region}: {self._launched_this_tick[region].name} and {cluster_type.name}")
        self._launched_this_tick[region] = cluster_type
        
        # Check availability
        if cluster_type == ClusterType.SPOT:
            if not self._spot_available_in_region(region):
                return False
        
        # Check if this is a cold start (no instances running anywhere)
        was_cold_start = len(self.active_instances) == 0 and self._last_active_region is None
        
        # Check if this is a region switch
        is_region_switch = False
        # Region switch: we had an active region before but now launching in a different region
        if self._last_active_region is not None and self._last_active_region != region:
            is_region_switch = True
            self._region_switch_info = {
                'from_region': self._last_active_region,
                'to_region': region,
                'tick': self.tick
            }
            logger.debug(f"Region switch detected: {self._last_active_region} -> {region} at tick {self.tick}")
            self.migration_count += 1
        
        # Launch successful - but don't overwrite if instance exists
        # We'll validate at the end that strategy cleaned up properly
        if region not in self.active_instances:
            self.active_instances[region] = cluster_type
        else:
            # Track that we need to validate this at tick end
            existing_type = self.active_instances[region]
            if not hasattr(self, '_pending_overwrites'):
                self._pending_overwrites = {}
            self._pending_overwrites[region] = (existing_type, cluster_type)
        # Immediately record cost for this tick
        self.current_tick_costs[region] = cluster_type
        # Mark that we had a new launch this tick
        # This should be set for ANY new launch: cold starts, region switches, OR same-region restarts
        self._new_launch_this_tick = True
        # Update last active region
        self._last_active_region = region
        logger.debug(f"Launched {cluster_type.name} in region {region} (cold_start={was_cold_start}, region_switch={is_region_switch})")
        return True
    
    def _terminate_internal(self, region: int):
        """Internal method to terminate an instance in a region."""
        if region in self.active_instances:
            ctype = self.active_instances.pop(region)
            # Remove from current tick costs to avoid charging if terminated at tick start
            if region in self.current_tick_costs:
                self.current_tick_costs.pop(region)
            logger.debug(f"Terminated {ctype.name} in region {region}")
            # Don't update _last_active_region on termination
            # This allows detecting region switches when terminate + launch in same tick
        else:
            logger.warning(f"No instance to terminate in region {region}")
    
    def _validate_no_multiple_instances(self):
        """Validate that strategy properly terminated instances when switching types."""
        if hasattr(self, '_pending_overwrites') and self._pending_overwrites:
            for region, (old_type, new_type) in self._pending_overwrites.items():
                if region in self.active_instances:
                    # Strategy launched new type but didn't terminate old type
                    logger.error(
                        f"Strategy error: Launched {new_type.name} in region {region} "
                        f"but failed to terminate existing {old_type.name} instance"
                    )
                    raise ValueError(
                        f"Strategy must explicitly terminate {old_type.name} in region {region} "
                        f"before launching {new_type.name}. Cannot have multiple instances in one region."
                    )
            # If we get here, strategy properly terminated (region not in active_instances)
            # So apply the pending launches
            for region, (_, new_type) in self._pending_overwrites.items():
                self.active_instances[region] = new_type
                self.current_tick_costs[region] = new_type
            self._pending_overwrites.clear()
    
    def _finalize_tick_costs(self):
        """Internal method to finalize costs for the current tick."""
        # Ensure all active instances are accounted for
        for region, ctype in self.active_instances.items():
            if region not in self.current_tick_costs:
                self.current_tick_costs[region] = ctype
        
        # Record to history
        if self.current_tick_costs:  # Only record if there are costs
            self.cost_history.append(self.current_tick_costs.copy())
        
        # Reset for next tick
        self.current_tick_costs = {}

    def get_constant_cost_map(self) -> Dict[ClusterType, float]:
        # Multi-region env doesn't have a single cost map
        raise NotImplementedError("Use envs[region_idx].get_constant_cost_map() for specific region")

    def get_price(self) -> Dict[ClusterType, float]:
        # Multi-region env doesn't have a single price
        raise NotImplementedError("Use envs[region_idx].get_price() for specific region")
    
    @property
    def accumulated_cost(self) -> float:
        """Calculate total accumulated cost across all regions."""
        total_cost = 0.0
        
        # Sum costs from all ticks in cost history
        for tick_costs in self.cost_history:
            for region, cluster_type in tick_costs.items():
                cost_map = self.envs[region].get_constant_cost_map()
                cost_per_hour = cost_map[cluster_type]
                total_cost += cost_per_hour * self.gap_seconds / 3600
                    
        return total_cost
    
    def get_cost_breakdown(self) -> Dict[str, typing.Any]:
        """Get detailed cost breakdown by region and type."""
        breakdown = {
            'total': self.accumulated_cost,
            'by_region': {},
            'by_type': {
                ClusterType.SPOT: 0.0,
                ClusterType.ON_DEMAND: 0.0,
                ClusterType.NONE: 0.0
            },
            'tick_count': len(self.cost_history)
        }
        
        # Calculate costs by region
        for region in range(self.num_regions):
            region_cost = 0.0
            for tick_costs in self.cost_history:
                if region in tick_costs:
                    cost_map = self.envs[region].get_constant_cost_map()
                    cost_per_hour = cost_map[tick_costs[region]]
                    region_cost += cost_per_hour * self.gap_seconds / 3600
            breakdown['by_region'][region] = region_cost
        
        # Calculate costs by type
        for tick_costs in self.cost_history:
            for region, cluster_type in tick_costs.items():
                cost_map = self.envs[region].get_constant_cost_map()
                cost_per_hour = cost_map[cluster_type]
                breakdown['by_type'][cluster_type] += cost_per_hour * self.gap_seconds / 3600
                
        return breakdown

    @property
    def config(self) -> dict:
        return {
            'name': self.NAME,
            'trace_files': self._trace_files,
            'env_start_hours': self.envs[0]._start_index * self.gap_seconds / 3600 if self.envs else 0,
            'gap_seconds': self.gap_seconds 
        }

    @classmethod
    def _from_args(
            cls,
            parser: 'configargparse.ArgumentParser') -> Sequence['MultiTraceEnv']:
        """Create MultiTraceEnv instances from command line arguments."""
        group = parser.add_argument_group('MultiTraceEnv')
        group.add_argument('--trace-files',
            type=str,
            nargs='+',
                           help='List of trace files for multi-region simulation')
        group.add_argument('--env-start-hours',
                           type=float,
                           default=0,
                           help='Start hours of the trace')
        args, _ = parser.parse_known_args()
        
        if hasattr(args, 'trace_files') and args.trace_files:
            return cls.create_env(args.trace_files, args.env_start_hours)
        else:
            # If no trace-files specified, fall back to single trace file if available
            if hasattr(args, 'trace_file') and args.trace_file:
                # For backward compatibility, treat single trace file as single region
                return [cls([args.trace_file], args.env_start_hours)]
            else:
                raise ValueError("Either --trace-files or --trace-file must be specified for multi_trace environment")
    
    @classmethod
    def create_env(cls, trace_files_or_dirs: List[str],
                   env_start_hours: float) -> Sequence['MultiTraceEnv']:
        """Create MultiTraceEnv instances from trace files or directories.
        
        If directories are provided, it will create one MultiTraceEnv with all json files from all directories.
        If files are provided, it will create one MultiTraceEnv with those files.
        """
        all_trace_files = []
        
        for path in trace_files_or_dirs:
            if os.path.isdir(path):
                # If it's a directory, add all json files from it
                for file in sorted(os.listdir(path)):
                    if file.endswith('.json'):
                        all_trace_files.append(os.path.join(path, file))
            else:
                # If it's a file, add it directly
                all_trace_files.append(path)
        
        if not all_trace_files:
            raise ValueError("No trace files found in the specified paths")
        
        # Return a single MultiTraceEnv with all trace files
        return [cls(all_trace_files, env_start_hours)]

    def update_strategy_progress(self, strategy: 'MultiRegionStrategy'):
        """Update task progress for multi-region strategies.
        
        This method replicates the task progress update logic from Strategy.step()
        for multi-region strategies that don't use step().
        """
        # Get information about which instances ran last tick
        active_instances = self.get_active_instances()
        
        logger.debug(f"update_strategy_progress: tick={self.tick}, active_instances={active_instances}")
        
        if not active_instances:
            # No instance running, no progress
            strategy.task_done_time.append(0)
            logger.debug("No active instances, appending 0 to task_done_time")
        else:
            # Check if we had a new launch in the previous tick
            if self._had_new_launch_last_tick:
                # First check if this was a region switch - these get special handling
                if self._region_switch_info is not None and self._region_switch_info['tick'] == self.tick - 1:
                    # Use dynamic migration model for region switches
                    from sky_spot.migration_model import get_migration_time_hours
                    
                    # Get region names
                    from_region_name = self.get_region_name(self._region_switch_info['from_region'])
                    to_region_name = self.get_region_name(self._region_switch_info['to_region'])
                    
                    # Get checkpoint size from task
                    checkpoint_size_gb = getattr(strategy.task, 'checkpoint_size_gb', 50.0)  # Default to 50GB
                    
                    # Get instance startup time from restart overhead
                    # For migration, use the restart overhead as the base startup time
                    instance_startup_hours = strategy.restart_overheads[0] / 3600.0  # Convert to hours
                    
                    # Calculate dynamic migration time
                    migration_hours = get_migration_time_hours(
                        from_region_name, to_region_name, checkpoint_size_gb,
                        instance_startup_hours=instance_startup_hours
                    )
                    # For region switches, we REPLACE the remaining overhead (not add to it)
                    # because the migration time includes the full cost of moving to the new region
                    strategy.remaining_restart_overhead = migration_hours * 3600  # Convert to seconds
                    logger.debug(f"Applied dynamic migration overhead: {migration_hours:.2f} hours "
                               f"({from_region_name} -> {to_region_name}, {checkpoint_size_gb}GB)")
                    
                    # Clear region switch info
                    self._region_switch_info = None
                else:
                    # New launch but NOT a region switch - cold start or same-region restart
                    # Apply standard restart overhead
                    current_subtask_index = 0
                    if hasattr(strategy, '_last_known_subtask_index'):
                        current_subtask_index = max(0, strategy._last_known_subtask_index)
                    restart_idx = min(current_subtask_index, len(strategy.restart_overheads) - 1)
                    strategy.remaining_restart_overhead = strategy.restart_overheads[restart_idx]
                    logger.debug(f"Applied restart overhead for new launch: {strategy.remaining_restart_overhead}s")
            
            # Calculate available time (one gap_second)
            available_time = self.gap_seconds
            
            # Account for restart overhead
            task_done_time = max(available_time - strategy.remaining_restart_overhead, 0)
            strategy.remaining_restart_overhead -= (available_time - task_done_time)
            if strategy.remaining_restart_overhead < 1e-3:
                strategy.remaining_restart_overhead = 0
            
            # Don't exceed remaining task time
            remaining_task_time = strategy.task_duration - sum(strategy.task_done_time)
            task_done_time = min(task_done_time, remaining_task_time)
            
            logger.debug(f"Appending task_done_time={task_done_time}, remaining_restart_overhead={strategy.remaining_restart_overhead}")
            strategy.task_done_time.append(task_done_time)
        
        # Handle ChainedTask updates (similar to Strategy.step())
        if hasattr(strategy, '_last_known_subtask_index') and hasattr(strategy.task, 'get_info'):
            task_info = strategy.task.get_info()
            current_subtask_index = task_info.get('current_sub_task_index', 0)
            if current_subtask_index > strategy._last_known_subtask_index:
                logger.debug(
                    f'Task changed from {strategy._last_known_subtask_index} to {current_subtask_index}'
                )
                strategy.restart_overhead = strategy.restart_overheads[current_subtask_index]
                strategy.inter_task_overhead = strategy.inter_task_overheads[current_subtask_index - 1]
                strategy._last_known_subtask_index = current_subtask_index
                strategy.remaining_restart_overhead = strategy.inter_task_overhead