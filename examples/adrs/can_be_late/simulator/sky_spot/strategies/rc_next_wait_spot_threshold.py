import argparse
import logging
import math
import numpy as np

from sky_spot.strategies import rc_threshold
from sky_spot.utils import ClusterType, COSTS

import typing
if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class RCNextWaitSpothresholdStrategy(rc_threshold.RCThresholdStrategy):
    NAME = 'rc_next_wait_spot_threshold'

    def __init__(self, args):
        args.keep_on_demand = None
        super().__init__(args)

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.cur_on_demand_time = 0

        self.cost_map = env.get_constant_cost_map()

    def _condition(self):
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0

    def _vm_to_idle_condition(self) -> bool:
        next_wait_length, next_spot_length = self.env.next_wait_spot_length()
        next_spot_time = next_spot_length * self.env.gap_seconds
        next_wait_time = next_wait_length * self.env.gap_seconds

        d = self.restart_overhead
        k = self.cost_map[ClusterType.ON_DEMAND] / self.cost_map[
            ClusterType.SPOT]

        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        c_0 = self.task_duration

        if next_wait_time <= (
                c_0 - c_t + next_spot_time - d
        ) * r_0 / c_0 - t - next_spot_time and next_spot_time > k * d / (k -
                                                                         1):
            return True
        return False

    def _idle_to_vm_condition(self) -> bool:
        # This should work with _condition()
        next_wait_length, next_spot_length = self.env.next_wait_spot_length()
        next_spot_time = next_spot_length * self.env.gap_seconds
        next_wait_time = next_wait_length * self.env.gap_seconds

        d = self.restart_overhead
        k = self.cost_map[ClusterType.ON_DEMAND] / self.cost_map[
            ClusterType.SPOT]

        if next_wait_time <= 2 * d and next_spot_time > k * d / (k - 1):
            return False
        return True

    def _vm_to_spot_condition(self) -> bool:
        assert self.env.spot_available()
        _, next_spot_length = self.env.next_wait_spot_length()
        next_spot_time = next_spot_length * self.env.gap_seconds
        d = self.restart_overhead
        k = self.cost_map[ClusterType.ON_DEMAND] / self.cost_map[
            ClusterType.SPOT]

        if next_spot_time > 2 * k * d / (k - 1):
            return True
        return False

    def _idle_to_spot_condition(self):
        assert self.env.spot_available()
        _, next_spot_length = self.env.next_wait_spot_length()
        next_spot_time = next_spot_length * self.env.gap_seconds
        d = self.restart_overhead
        k = self.cost_map[ClusterType.ON_DEMAND] / self.cost_map[
            ClusterType.SPOT]

        if next_spot_time > k * d / (k - 1):
            return True
        return False

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        env = self.env

        # Make decision for the gap starting from env.tick
        remaining_time = math.floor((self.deadline - env.elapsed_seconds) /
                                    env.gap_seconds) * env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE

        current_cluster_type = env.cluster_type
        if current_cluster_type == ClusterType.NONE:
            if has_spot and self._idle_to_spot_condition():
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.NONE
                if self._condition() < 0 and self._idle_to_vm_condition():
                    request_type = ClusterType.ON_DEMAND
        elif current_cluster_type == ClusterType.ON_DEMAND:
            if has_spot and self._vm_to_spot_condition():
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.ON_DEMAND
                if self._vm_to_idle_condition():
                    request_type = ClusterType.NONE
        else:
            assert current_cluster_type == ClusterType.SPOT
            if has_spot:
                request_type = ClusterType.SPOT
            else:
                request_type = ClusterType.NONE
                if self._condition() < 0:
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
            cls, parser: 'argparse.ArgumentParser'
    ) -> 'RCNextWaitSpothresholdStrategy':
        group = parser.add_argument_group('RCV2DThresholdStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
