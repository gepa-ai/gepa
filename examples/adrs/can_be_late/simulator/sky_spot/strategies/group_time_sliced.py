import argparse
import logging
import math
import typing

from sky_spot.strategies import time_sliced
from sky_spot.utils import ClusterType
from sky_spot import task

if typing.TYPE_CHECKING:
    from sky_spot import env

logger = logging.getLogger(__name__)


class GroupTimeSlicedStrategy(time_sliced.TimeSlicedStrategy):
    NAME = 'group_time_sliced'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        args = self.args

        self.slice_groups = args.slice_interval_hours_groups

        slice_group_seconds = [t * 3600 for t in self.slice_groups]

        self.slice_intervals = []
        per_group_time = self.deadline / len(slice_group_seconds)
        group_id = 0
        while sum(self.slice_intervals) < self.deadline:
            self.slice_intervals.append(slice_group_seconds[group_id])
            if sum(self.slice_intervals) > per_group_time * (group_id + 1):
                group_id += 1

        self.slice_intervals[-1] = self.deadline - sum(
            self.slice_intervals[:-1])

        self.num_slices = len(self.slice_intervals)

        self.use_avg_gain = args.use_avg_gain

        self.slice_task_durations = [
            self.task_duration * interval / self.deadline
            for interval in self.slice_intervals
        ]
        self.slice_gap_counts = None
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

    @classmethod
    def _from_args(
            cls,
            parser: 'argparse.ArgumentParser') -> 'GroupTimeSlicedStrategy':
        group = parser.add_argument_group('GroupTimeSlicedStrategy')
        group.add_argument('--slice-interval-hours-groups',
                           type=int,
                           nargs='+',
                           default=[1])
        group.add_argument('--use-avg-gain', action='store_true')
        args, _ = parser.parse_known_args()
        return cls(args)

    @property
    def name(self):
        use_avg_str = '_avg' if self.use_avg_gain else ''
        return f'{self.NAME}{use_avg_str}_{"_".join(map(str, self.slice_groups))}h'

    @property
    def config(self):
        return dict(
            super().config,
            num_pairs=self.num_slices,
            slice_interval_hours_groups=self.slice_groups,
            slice_intervals=self.slice_intervals,
            slice_task_durations=self.slice_task_durations,
            pair_gap_counts=self.slice_gap_counts,
            use_avg_gain=self.use_avg_gain,
        )
