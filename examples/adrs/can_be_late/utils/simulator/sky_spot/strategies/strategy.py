import json
import logging
import typing
import math

from sky_spot.utils import ClusterType
from sky_spot import task as task_lib

if typing.TYPE_CHECKING:
    import configargparse
    from sky_spot import env
    from sky_spot.multi_region_types import Action, LaunchResult

logger = logging.getLogger(__name__)


class Strategy:
    NAME = 'abstract'
    SUBCLASSES: typing.Dict[str, typing.Type['Strategy']] = {}

    def __init__(self, args):
        self.args = args

    def reset(self, env: 'env.Env', task: 'task_lib.Task'):
        args = self.args
        self.task = task

        self.deadline = args.deadline_hours * 3600
        self.task_duration = self.task.get_total_duration_seconds()

        self.restart_overheads: list[float] = [
            oh * 3600 for oh in args.restart_overhead_hours
        ]
        self.restart_overhead: float = self.restart_overheads[0]

        self.inter_task_overheads: list[float] = [
            oh * 3600 for oh in getattr(args, 'inter_task_overhead', [0.0])
        ]
        self.inter_task_overhead: float = self.inter_task_overheads[0]

        self.task_done_time: list[float] = []
        self.task.set_progress_source(self.task_done_time)

        self.remaining_restart_overhead: float = 0.0
        # Initialize inter-task overhead state
        self.remaining_inter_task_overhead = 0
        self._last_known_subtask_index = -1
        if isinstance(self.task, task_lib.ChainedTask):
            task_info = self.task.get_info()
            self._last_known_subtask_index = task_info.get(
                'current_sub_task_index', 0)

        self.env: 'env.Env' = env

    def __init_subclass__(cls):
        assert cls.NAME not in cls.SUBCLASSES and cls.NAME != 'abstract', f'Name {cls.NAME} already exists'
        cls.SUBCLASSES[cls.NAME] = cls

    def __repr__(self) -> str:
        return f'{self.NAME}({json.dumps(self.config)})'

    def _apply_strong_guarantee(self, proposed_decision: ClusterType,
                                last_cluster_type: ClusterType) -> ClusterType:
        """The non-negotiable final check."""
        remaining_time = math.floor(
            (self.deadline - self.env.elapsed_seconds) /
            self.env.gap_seconds) * self.env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)

        # Task is already finished, no need for guarantee.
        if remaining_task_time <= 1e-3:
            return proposed_decision

        total_task_remaining = math.ceil(
            (remaining_task_time + self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        if total_task_remaining >= remaining_time:
            # If we are already on a working spot instance, we can risk it for one more step
            if last_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                return ClusterType.SPOT
            else:
                # In all other critical situations, force ON_DEMAND
                if proposed_decision != ClusterType.ON_DEMAND:
                    logger.warning(
                        f"Tick {self.env.tick}: STRONG GUARANTEE OVERRIDE. "
                        f"Proposed: {proposed_decision.name}, Final: ON_DEMAND")
                return ClusterType.ON_DEMAND

        # If the strong guarantee is not triggered, respect the subclass's decision
        return proposed_decision

    def _before_decision_hook(self, last_cluster_type: ClusterType,
                              request_type: ClusterType):
        """A hook for subclasses to add logic before the final decision."""
        pass

    def step(self) -> ClusterType:
        # Realize the information of the last gap
        env = self.env

        try:
            last_cluster_type, has_spot = env.observe()
        except ValueError as exc:
            logger.warning(
                'env.observe() failed with %s; treating as no spot availability '
                'and continuing safely',
                exc,
            )
            last_cluster_type, has_spot = ClusterType.NONE, False
        if last_cluster_type == ClusterType.NONE:
            self.task_done_time.append(0)
        else:
            available_time = env.gap_seconds
            task_done_time = max(
                available_time - self.remaining_restart_overhead, 0)
            self.remaining_restart_overhead -= (available_time -
                                                task_done_time)
            if self.remaining_restart_overhead < 1e-3:
                self.remaining_restart_overhead = 0

            remaining_task_time = self.task_duration - sum(self.task_done_time)
            task_done_time = min(task_done_time, remaining_task_time)
            self.task_done_time.append(task_done_time)

        # Let the subclass make its heuristic decision
        heuristic_decision = self._step(last_cluster_type, has_spot)

        # Apply the STRONG GUARANTEE as a final check
        request_type = self._apply_strong_guarantee(heuristic_decision,
                                                    last_cluster_type)
        if request_type != heuristic_decision:
            logger.warning(
                f'{env.tick}: STRONG GUARANTEE OVERRIDE. '
                f'Proposed: {heuristic_decision.name}, Final: {request_type.name}'
            )

        # Allow subclasses to add logic before the final decision is made
        self._before_decision_hook(last_cluster_type, request_type)

        # Final safety check: ensure SPOT is actually available in current region
        if request_type == ClusterType.SPOT and not env.spot_available():
            logger.warning(
                f"Tick {env.tick}: SAFETY OVERRIDE - SPOT requested but not available in current region. "
                f"Changing to NONE."
            )
            request_type = ClusterType.NONE

        task_changed = False
        current_subtask_index = 0
        if isinstance(self.task, task_lib.ChainedTask):
            task_info = self.task.get_info()
            current_subtask_index = task_info['current_sub_task_index']
            if current_subtask_index > self._last_known_subtask_index:
                logger.debug(
                    f'Task changed from {self._last_known_subtask_index} to {current_subtask_index}'
                )
                task_changed = True
                self.restart_overhead = self.restart_overheads[
                    current_subtask_index]

                # (subtask index - 1) corresponds to the transition overhead between tasks
                self.inter_task_overhead = self.inter_task_overheads[
                    current_subtask_index - 1]
                self._last_known_subtask_index = current_subtask_index

        if task_changed:
            self.remaining_restart_overhead = self.inter_task_overhead

        current_cluster_type = last_cluster_type
        if last_cluster_type == ClusterType.SPOT and not has_spot:
            current_cluster_type = ClusterType.NONE
        if current_cluster_type != request_type and request_type != ClusterType.NONE:
            if not task_changed:
                # Get the current task's restart overhead
                restart_idx = min(current_subtask_index,
                                  len(self.restart_overheads) - 1)
                # ! Should be low priority than inter-task overhead
                self.remaining_restart_overhead = self.restart_overheads[
                    restart_idx]
        return request_type

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        raise NotImplementedError

    @property
    def task_done(self):
        return self.task.is_done

    @property
    def config(self):
        return {
            'name': self.NAME,
            'deadline': self.deadline,
            'task_duration': self.task_duration,
            'restart_overhead': self.restart_overhead,
            'restart_overheads': self.restart_overheads,
            'inter_task_overhead': self.inter_task_overhead,
            'inter_task_overheads': self.inter_task_overheads,
            'task': self.task.get_config(),
        }

    @classmethod
    def from_args(cls, parser: 'configargparse.ArgumentParser') -> 'Strategy':
        # parser.add_argument(f'--strategy-config', type=str, default=None, is_config_file=True, required=False)
        parser.add_argument('--strategy',
                            type=str,
                            default='strawman',
                            choices=cls.SUBCLASSES.keys())
        parser.add_argument(
            '--inter-task-overhead',
            type=float,
            default=[0.0],
            nargs='+',
            help=
            'Overhead in hours incurred when switching between sub-tasks in a ChainedTask.'
        )
        args, _ = parser.parse_known_args()
        cls = cls.SUBCLASSES[args.strategy]
        return cls._from_args(parser)

    def info(self):
        assert self.task is not None
        task_info = self.task.get_info()
        prefixed_task_info = {f'Task/{k}': v for k, v in task_info.items()}

        current_overhead_idx = 0
        current_restart_idx = 0
        if isinstance(self.task, task_lib.ChainedTask):
            current_subtask_index = task_info.get('current_sub_task_index', 0)
            if current_subtask_index > 0:
                current_overhead_idx = min(current_subtask_index - 1,
                                           len(self.inter_task_overheads) - 1)
            current_restart_idx = min(current_subtask_index,
                                      len(self.restart_overheads) - 1)

        strategy_info = {
            'Strategy/RemainingRestartOverhead(seconds)':
            self.remaining_restart_overhead,
            'Strategy/CurrentRestartOverheadIndex':
            current_restart_idx,
            'Strategy/CurrentInterTaskOverheadIndex':
            current_overhead_idx,
            'Strategy/RemainingInterTaskOverhead(seconds)':
            self.remaining_inter_task_overhead
        }
        return {**prefixed_task_info, **strategy_info}

    @classmethod
    def get(cls, name: str) -> type['Strategy']:
        return cls.SUBCLASSES[name]

    @property
    def name(self) -> str:
        return self.NAME

    @classmethod
    def _from_args(cls, parser: 'configargparse.ArgumentParser') -> 'Strategy':
        raise NotImplementedError


class MultiRegionStrategy(Strategy):
    """Base class for multi-region strategies using yield/generator pattern."""
    
    # Override NAME to avoid conflict with parent class registration
    NAME = 'multi_region_base'
    
    def __init_subclass__(cls):
        # Don't register the base multi-region class
        if cls.NAME != 'multi_region_base':
            super().__init_subclass__()
    
    def step(self) -> ClusterType:
        """Not used for multi-region strategies."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is a multi-region strategy and should use _step_multi() instead"
        )
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        """Not used for multi-region strategies."""
        # This should never be called for multi-region strategies
        raise NotImplementedError(
            f"{self.__class__.__name__} is a multi-region strategy and should use _step_multi() instead"
        )
    
    def _step_multi(self) -> typing.Generator['Action', typing.Optional['LaunchResult'], None]:
        """Multi-region strategy interface using yield/generator pattern.
        
        Yields:
            Action: The action to perform (TryLaunch or Terminate)
            
        Receives:
            Optional[LaunchResult]: Result of TryLaunch action (None for Terminate)
            
        Example:
            # Try to launch in region 0
            result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
            if not result.success:
                # Try region 1 if region 0 failed
                result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
        """
        raise NotImplementedError
