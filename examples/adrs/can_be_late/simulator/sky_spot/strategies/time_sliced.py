import argparse
import logging
import math
import typing

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class TimeSlicedStrategy(strategy.Strategy):
    NAME = 'time_sliced'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        args = self.args
        slice_interval = args.slice_interval_hours * 3600
        self.slice_interval = slice_interval
        self.num_slices = math.ceil(self.deadline / slice_interval)

        self.use_avg_gain = args.use_avg_gain

        self.slice_intervals = [slice_interval for _ in range(self.num_slices)]
        self.slice_intervals[-1] = self.deadline - sum(
            self.slice_intervals[:-1])
        self.slice_task_durations = [
            self.task_duration * interval / self.deadline
            for interval in self.slice_intervals
        ]
        self.slice_index = 0

        self.previous_gain_seconds = 0
        self.avg_gain = 0

        self.slice_gap_counts = [
            int(round(interval / env.gap_seconds, 0))
            for interval in self.slice_intervals
        ]
        for i, slice_gap_counts in enumerate(self.slice_gap_counts):
            assert abs(slice_gap_counts * env.gap_seconds -
                       self.slice_intervals[i]) < 1e-4, (
                           slice_gap_counts, env.gap_seconds,
                           self.slice_intervals[i])

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        env = self.env
        # Make decision for the gap starting from env.tick
        slice_end_gap_index = sum(self.slice_gap_counts[:self.slice_index + 1])
        if slice_end_gap_index <= len(self.task_done_time) - 1:
            self.slice_index += 1
            if self.slice_index >= self.num_slices:
                delta = self.task_duration - sum(self.task_done_time)
                assert delta <= env.gap_seconds, delta
                self.task_done_time.append(delta)
                return ClusterType.NONE
            last_slice_gap_counts = self.slice_gap_counts[self.slice_index - 1]
            last_slice_gain = sum(
                self.task_done_time[-last_slice_gap_counts:]
            ) - self.slice_task_durations[self.slice_index - 1]
            self.previous_gain_seconds += last_slice_gain
            logger.debug(
                f'==> {self.env.tick}: Pair {self.slice_index} starts (last gain: {last_slice_gain/3600:.2f}, previous_gain: {self.previous_gain_seconds/3600:.2f})'
            )
            logger.debug(
                f'==> Task done time: {sum(self.task_done_time)/3600:.2f}')
            self.avg_gain = self.previous_gain_seconds / (self.num_slices -
                                                          self.slice_index)
        if self.slice_index >= self.num_slices:
            delta = self.task_duration - sum(self.task_done_time)
            assert delta < 1e-2, delta
            return ClusterType.NONE

        assert self.slice_index < self.num_slices, ('Pair index out of range',
                                                    self.slice_index,
                                                    self.num_slices)

        slice_start_gap_index = sum(self.slice_gap_counts[:self.slice_index])
        slice_end_gap_index = sum(self.slice_gap_counts[:self.slice_index + 1])
        slice_end_seconds = slice_end_gap_index * self.env.gap_seconds

        slice_remaining_time = slice_end_seconds - env.elapsed_seconds
        remaining_task_time = self.slice_task_durations[
            self.slice_index] - sum(
                self.task_done_time[slice_start_gap_index + 1:])
        total_task_remaining = self.task_duration - sum(self.task_done_time)
        if total_task_remaining <= 1e-3:
            return ClusterType.NONE

        if has_spot:
            request_type = ClusterType.SPOT
        else:
            request_type = ClusterType.NONE

        switch_task_remaining = (remaining_task_time + self.restart_overhead)
        switch_task_remaining = math.ceil(
            switch_task_remaining /
            self.env.gap_seconds) * self.env.gap_seconds

        switch_task_remaining_with_2d = math.ceil(
            (remaining_task_time + 2 * self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        gain_seconds = self.avg_gain if self.use_avg_gain else self.previous_gain_seconds
        pair_available_time = slice_remaining_time + gain_seconds

        pair_available_time = math.floor(
            pair_available_time / self.env.gap_seconds) * self.env.gap_seconds

        current_cluster_type = env.cluster_type
        total_task_remaining = math.ceil(
            (self.task_duration - sum(self.task_done_time) +
             self.restart_overhead) / env.gap_seconds) * env.gap_seconds
        total_task_remaining_with_2d = math.ceil(
            (self.task_duration - sum(self.task_done_time) +
             2 * self.restart_overhead) / env.gap_seconds) * env.gap_seconds

        deadline_remaining_time = math.floor(
            (self.deadline - env.elapsed_seconds) /
            env.gap_seconds) * env.gap_seconds

        if ((remaining_task_time - gain_seconds) <= 1e-3
                and total_task_remaining_with_2d < deadline_remaining_time):
            return request_type

        if (current_cluster_type in [ClusterType.NONE, ClusterType.ON_DEMAND]
                and
            (switch_task_remaining_with_2d >= pair_available_time
             or total_task_remaining_with_2d >= deadline_remaining_time)):
            request_type = ClusterType.ON_DEMAND

        if switch_task_remaining >= pair_available_time or total_task_remaining >= deadline_remaining_time:
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                assert has_spot
                logger.debug(
                    f'{env.tick}: Deadline reached, stay on spot '
                    f'(task remaining: {(self.task_duration - sum(self.task_done_time))/3600:.2f}, pair available: {pair_available_time/3600:.2f})'
                )
                # Keep the spot VM until preemption
                request_type = ClusterType.SPOT
            else:
                logger.debug(
                    f'{env.tick}: Deadline reached, switch to on-demand '
                    f'(task remaining: {(self.task_duration - sum(self.task_done_time))/3600:.2f}, pair available: {pair_available_time/3600:.2f})'
                )
                # We need to finish it on time by switch to on-demand
                request_type = ClusterType.ON_DEMAND
        else:
            logger.debug(
                f'{env.tick}: {request_type}'
                f'(task remaining: {(self.task_duration - sum(self.task_done_time))/3600:.2f}, pair available: {pair_available_time/3600:.2f})'
            )
        return request_type

    @classmethod
    def _from_args(cls,
                   parser: 'argparse.ArgumentParser') -> 'TimeSlicedStrategy':
        group = parser.add_argument_group('PairAmortizeStrategy')
        group.add_argument('--slice-interval-hours', type=int, default=1)
        group.add_argument('--use-avg-gain', action='store_true')
        args, _ = parser.parse_known_args()
        return cls(args)

    @property
    def name(self):
        use_avg_str = '_avg' if self.use_avg_gain else ''
        return f'{self.NAME}{use_avg_str}_{self.slice_interval/3600}h'

    @property
    def config(self):
        return {
            **super().config,
            'num_pairs': self.num_slices,
            'slice_intervals': self.slice_intervals,
            'slice_task_durations': self.slice_task_durations,
            'pair_gap_counts': self.slice_gap_counts,
            'use_avg_gain': self.use_avg_gain,
        }
