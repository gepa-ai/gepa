import argparse
import math
import typing
import logging
from sky_spot.utils import ClusterType

from sky_spot.strategies import time_sliced

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class LooseTimeSlicedStrategy(time_sliced.TimeSlicedStrategy):
    NAME = 'loose_time_sliced'

    def __init__(self, args):
        super().__init__(args)

        self.max_total_slacks = args.max_total_slacks
        self.max_slice_slacks = args.max_slice_slacks
        self.total_slacks = 0
        self.slice_slacks = 0

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)
        self.total_slacks = 0
        self.slice_slacks = 0

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        env = self.env
        # Make decision for the gap starting from env.tick
        slice_end_gap_index = sum(self.slice_gap_counts[:self.slice_index + 1])
        if slice_end_gap_index <= len(self.task_done_time) - 1:
            self.slice_index += 1
            self.slice_slacks = 0
            if self.slice_index >= self.num_slices:
                delta = self.task_duration - sum(self.task_done_time)
                assert delta <= env.gap_seconds, (delta, self.config)
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
        total_task_remaining_time = self.task_duration - sum(
            self.task_done_time)
        if total_task_remaining_time <= 1e-3:
            return ClusterType.NONE

        if has_spot:
            request_type = ClusterType.SPOT
        else:
            request_type = ClusterType.NONE

        switch_task_remaining = (remaining_task_time + self.restart_overhead)
        switch_task_remaining = math.ceil(
            switch_task_remaining /
            self.env.gap_seconds) * self.env.gap_seconds
        gain_seconds = self.avg_gain if self.use_avg_gain else self.previous_gain_seconds
        pair_available_time = slice_remaining_time + gain_seconds
        pair_available_time = math.floor(
            pair_available_time / self.env.gap_seconds) * self.env.gap_seconds

        total_task_remaining = math.ceil(
            (self.task_duration - sum(self.task_done_time) +
             self.restart_overhead) / env.gap_seconds) * env.gap_seconds
        deadline_remaining = math.floor((self.deadline - env.elapsed_seconds) /
                                        env.gap_seconds) * env.gap_seconds
        total_task_remaining_with_2d = math.ceil(
            (total_task_remaining_time + 2 * self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        current_cluster_type = env.cluster_type
        if switch_task_remaining >= pair_available_time:
            # For previous slices, we can switch between spot and on-demand freely, as we don't need a strong guarantee for the deadline.
            if has_spot:
                logger.debug(
                    f'{env.tick}: Deadline reached, but we use loose strategy, so we still use spot '
                    f'(task remaining: {(self.task_duration - sum(self.task_done_time))/3600:.2f}, pair available: {pair_available_time/3600:.2f})'
                )
                if current_cluster_type == ClusterType.ON_DEMAND:
                    self.total_slacks += 1
                    self.slice_slacks += 1
                request_type = ClusterType.SPOT
            else:
                # We need to finish it on time by switch to on-demand
                request_type = ClusterType.ON_DEMAND

            if self.max_total_slacks is not None and self.total_slacks > self.max_total_slacks:
                logger.debug(f'Exceed max total slacks: {self.total_slacks}')
                request_type = ClusterType.ON_DEMAND
            if self.max_slice_slacks is not None and self.slice_slacks > self.max_slice_slacks:
                logger.debug(f'Exceed max slice slacks: {self.slice_slacks}')
                request_type = ClusterType.ON_DEMAND

        if (current_cluster_type == ClusterType.NONE
                and total_task_remaining_with_2d >= deadline_remaining):
            request_type = ClusterType.ON_DEMAND
        if total_task_remaining >= deadline_remaining:
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                logger.debug(
                    f'{env.tick}: Deadline reached, but  '
                    f'(task remaining: {(self.task_duration - sum(self.task_done_time))/3600:.2f}, pair avilable: {pair_available_time/3600:.2f})'
                )
                request_type = ClusterType.SPOT
            else:
                # We need to finish it on time by switch to on-demand
                request_type = ClusterType.ON_DEMAND

        return request_type

    @classmethod
    def _from_args(
            cls,
            parser: 'argparse.ArgumentParser') -> 'LooseTimeSlicedStrategy':
        group = parser.add_argument_group('LooseTimeSlicedStrategy')
        group.add_argument('--max-total-slacks', type=int, default=None)
        group.add_argument('--max-slice-slacks', type=int, default=None)
        group.add_argument('--slice-interval-hours', type=int, default=1)
        group.add_argument('--use-avg-gain', action='store_true')
        args, _ = parser.parse_known_args()
        return cls(args)

    @property
    def name(self):
        use_avg_str = '_avg' if self.use_avg_gain else ''
        name = f'{self.NAME}{use_avg_str}_{self.slice_interval/3600}h'
        if self.max_total_slacks is not None:
            name += f'_ts_{self.max_total_slacks}'
        if self.max_slice_slacks is not None:
            name += f'_ss_{self.max_slice_slacks}'
        return name

    @property
    def config(self):
        return dict(
            super().config,
            num_pairs=self.num_slices,
            slice_intervals=self.slice_intervals,
            slice_task_durations=self.slice_task_durations,
            pair_gap_counts=self.slice_gap_counts,
            use_avg_gain=self.use_avg_gain,
            max_total_slacks=self.max_total_slacks,
            max_slice_slacks=self.max_slice_slacks,
        )
