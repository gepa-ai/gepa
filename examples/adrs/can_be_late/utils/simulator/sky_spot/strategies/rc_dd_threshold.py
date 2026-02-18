import argparse
import logging
import math

from sky_spot.strategies import rc_threshold
from sky_spot.utils import ClusterType

logger = logging.getLogger(__name__)


class RCDDThresholdStrategy(rc_threshold.RCThresholdStrategy):
    NAME = 'rc_dd_threshold'

    def __init__(self, args):
        super().__init__(args)
        self.keep_on_demand = args.keep_on_demand
        self.dd_scale = args.dd_scale

    def reset(self, env):
        super().reset(env)
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
        t_0 = self.deadline
        c_0 = self.task_duration

        return self.dd_scale * d / self.cur_on_demand_time - (t_0 - c_0) / c_0

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        env = self.env

        # Make decision for the gap starting from env.tick
        remaining_time = math.floor((self.deadline - env.elapsed_seconds) /
                                    env.gap_seconds) * env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        if has_spot:
            request_type = ClusterType.SPOT
        else:
            request_type = ClusterType.NONE

        current_cluster_type = env.cluster_type
        if self._condition() < 0:
            if has_spot:
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.ON_DEMAND

        if current_cluster_type == ClusterType.ON_DEMAND:
            if self.keep_on_demand:
                if not has_spot or self._condition2() >= 0:
                    # Keep the on-demand VM until the end
                    logger.debug(f'{env.tick}: Keep on-demand VM')
                    request_type = ClusterType.ON_DEMAND
            else:
                if not has_spot and self._condition2() >= 0:
                    # Keep the on-demand VM until the end
                    logger.debug(f'{env.tick}: Keep on-demand VM')
                    request_type = ClusterType.ON_DEMAND

        total_task_remaining = math.ceil(
            (remaining_task_time + self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        total_task_remaining_with_2D = math.ceil(
            (remaining_task_time + 2 * self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds
        if total_task_remaining_with_2D >= remaining_time and current_cluster_type in [
                ClusterType.NONE, ClusterType.ON_DEMAND
        ]:
            # We need to switch to on-demand when it is T+2D
            logger.debug(f'{env.tick}: Deadline reached, switch to on-demand')
            request_type = ClusterType.ON_DEMAND

        if total_task_remaining >= remaining_time:
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                # Keep the spot VM until preemption
                logger.debug(
                    f'{env.tick}: Deadline reached, keep spot until preemption'
                )
                request_type = ClusterType.SPOT
            else:
                logger.debug(
                    f'{env.tick}: Deadline reached, switch to on-demand')
                # We need to finish it on time by switch to on-demand
                request_type = ClusterType.ON_DEMAND
            if self.restart_overhead == 0 and has_spot:
                # We can switch to spot without cost.
                request_type = ClusterType.SPOT

        if request_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time += self.env.gap_seconds
        elif last_cluster_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time = 0

        return request_type

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser') -> 'RCDDThresholdStrategy':
        group = parser.add_argument_group('RCThresholdStrategy')
        group.add_argument('--keep-on-demand', action='store_true')
        group.add_argument('--dd-scale', type=float, default=1.0)
        args, _ = parser.parse_known_args()
        return cls(args)
