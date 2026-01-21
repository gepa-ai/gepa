import argparse
import logging
import math

import numpy as np

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

import typing
if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class RCLWThresholdStrategy(strategy.Strategy):
    NAME = 'rc_lw_threshold'

    # Update reset signature and super call
    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)

        trace = env.get_trace_before_end(self.deadline)
        avail_trace = 1 - np.array(trace)

        def consecutive_1s(trace):
            padded_trace = np.array([0] + trace.tolist() + [0])
            diff = padded_trace[1:] - padded_trace[:-1]
            lengths = np.where(diff == -1)[0] - np.where(diff == 1)[0]
            # discard the first and last 1s, as they may not be complete
            lengths = lengths[1:-1]
            return lengths

        # life_times = consecutive_1s(avail_trace)
        # self.avg_life_time = np.mean(life_times)
        # self.std_life_time = np.std(life_times)

        wait_times = consecutive_1s(1 - avail_trace)
        self.avg_wait_time = np.mean(wait_times) if len(wait_times) > 0 else 0
        self.std_wait_time = np.std(wait_times) if len(wait_times) > 0 else 0

        self.last_vm_timestamp = 0
        self.last_spot_timestamp = 0
        self.last_task_done_time = 0

    def _condition(self):
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0

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
            switch_back_time = (self.task_duration + self.last_vm_timestamp -
                                self.last_task_done_time -
                                self.task_duration / self.deadline *
                                (self.last_spot_timestamp +
                                 self.avg_wait_time + self.std_wait_time) +
                                5 * self.restart_overhead)
            if not has_spot:
                if env.elapsed_seconds >= switch_back_time:
                    request_type = ClusterType.NONE
                else:
                    request_type = ClusterType.ON_DEMAND
            elif env.elapsed_seconds - self.last_vm_timestamp < 3 * self.deadline / (
                    self.deadline -
                    self.task_duration) * self.restart_overhead:
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

        if current_cluster_type != ClusterType.ON_DEMAND and request_type == ClusterType.ON_DEMAND:
            self.last_vm_timestamp = env.elapsed_seconds
            self.last_task_done_time = sum(self.task_done_time)
        if has_spot:
            self.last_spot_timestamp = env.elapsed_seconds
        return request_type

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser') -> 'RCLWThresholdStrategy':
        group = parser.add_argument_group('RCThresholdStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
