import argparse
import logging
import math
import typing

from sky_spot.strategies import time_sliced
from sky_spot.strategies import loose_time_sliced

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class LooseTimeSlicedByNumStrategy(loose_time_sliced.LooseTimeSlicedStrategy):
    NAME = 'loose_time_sliced_by_num'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.total_slacks = 0
        self.slice_slacks = 0

        args = self.args
        self.num_slices = args.num_slices
        self.use_avg_gain = args.use_avg_gain

        self.slice_intervals = []
        slice_interval = self.deadline / self.num_slices
        for i in range(self.num_slices):
            actual_slice_interval = (i + 1) * slice_interval - sum(
                self.slice_intervals)
            rounded_slice_interval = math.floor(
                actual_slice_interval / env.gap_seconds) * env.gap_seconds
            self.slice_intervals.append(rounded_slice_interval)
        self.slice_intervals[-1] = self.deadline - sum(
            self.slice_intervals[:-1])
        # Note: the sum of slice_intervals may be slightly smaller than the deadline by at most 1 gap
        self.slice_task_durations = [
            self.task_duration * interval / self.deadline
            for interval in self.slice_intervals
        ]
        remaining_task = self.task_duration - sum(self.slice_task_durations)
        for i in range(self.num_slices):
            self.slice_task_durations[i] += remaining_task / self.num_slices
        self.slice_gap_counts = None
        self.slice_index = 0

        self.previous_gain_seconds = 0
        self.avg_gain = 0

        self.slice_gap_counts = [
            int(round(interval / env.gap_seconds, 0))
            for interval in self.slice_intervals
        ]
        self.slice_gap_counts[-1] = int(self.slice_intervals[-1] /
                                        env.gap_seconds)
        for i, slice_gap_counts in enumerate(self.slice_gap_counts[:-1]):
            assert abs(slice_gap_counts * env.gap_seconds -
                       self.slice_intervals[i]) < 1e-4, (
                           slice_gap_counts, env.gap_seconds,
                           self.slice_intervals[i])

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser'
    ) -> 'LooseTimeSlicedByNumStrategy':
        group = parser.add_argument_group('LooseTimeSlicedByNumStrategy')
        group.add_argument('--num-slices', type=int, default=1)
        group.add_argument('--use-avg-gain', action='store_true')
        group.add_argument('--max-total-slacks', type=int, default=None)
        group.add_argument('--max-slice-slacks', type=int, default=None)
        group.add_argument('--use-avg-gain', action='store_true')
        args, _ = parser.parse_known_args()
        return cls(args)

    @property
    def name(self):
        use_avg_str = '_avg' if self.use_avg_gain else ''
        return f'{self.NAME}{use_avg_str}_n={self.num_slices}_ts={self.max_total_slacks}_ss={self.max_slice_slacks}'
