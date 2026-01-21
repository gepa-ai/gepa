import argparse
import math
import typing
import random

from sky_spot.strategies import time_sliced

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


class RandomTimeSlicedStrategy(time_sliced.TimeSlicedStrategy):
    NAME = 'random_time_sliced'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        args = self.args
        random.seed(args.slice_selection_seed)
        slice_interval = [t * 3600 for t in args.slice_interval_hours_choices]
        self.slice_interval = slice_interval

        max_num_slices = math.ceil(self.deadline / min(slice_interval))
        random_slice_intervals = [
            random.choice(slice_interval) for _ in range(max_num_slices)
        ]
        cnt = 0
        while sum(random_slice_intervals[:cnt]) < self.deadline:
            cnt += 1
        self.num_slices = cnt

        self.use_avg_gain = args.use_avg_gain

        self.slice_intervals = random_slice_intervals[:self.num_slices]
        self.slice_intervals[-1] = self.deadline - sum(
            self.slice_intervals[:-1])
        self.slice_task_durations = [
            self.task_duration * interval / self.deadline
            for interval in self.slice_intervals
        ]
        self.slice_gap_counts = None
        self.slice_index = 0

        self.previous_gain_seconds = 0
        self.avg_gain = 0

    @classmethod
    def _from_args(
            cls,
            parser: 'argparse.ArgumentParser') -> 'RandomTimeSlicedStrategy':
        group = parser.add_argument_group('Random Time Sliced Strategy')
        group.add_argument('--slice-selection-seed', type=int, default=0)
        group.add_argument('--slice-interval-hours-choices',
                           type=int,
                           nargs='+',
                           default=[2, 4, 8])
        group.add_argument('--use-avg-gain', action='store_true')
        args, _ = parser.parse_known_args()
        return cls(args)

    @property
    def name(self):
        use_avg_str = '_avg' if self.use_avg_gain else ''
        return f'{self.NAME}{use_avg_str}_{"_".join([interval/3600 for interval in self.slice_interval])}'

    @property
    def config(self):
        return dict(
            super().config,
            num_pairs=self.num_slices,
            slice_intervals=self.slice_intervals,
            slice_task_durations=self.slice_task_durations,
            pair_gap_counts=self.slice_gap_counts,
            use_avg_gain=self.use_avg_gain,
        )
