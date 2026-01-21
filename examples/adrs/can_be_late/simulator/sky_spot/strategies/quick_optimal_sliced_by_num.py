import argparse
import math
import multiprocessing
import time
import typing

import pulp

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType, COSTS

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


class QuickOptimalSlicedByNumStrategy(strategy.Strategy):
    NAME = 'quick_optimal_sliced_by_num'

    def get_spot_intervals(self, preempt_list):
        # return a list of (spot_start_time, spot_life_time)
        ret = []
        st = 0
        while True:
            # find the first 1
            for i in range(st, len(preempt_list)):
                if preempt_list[i] == 0:
                    st = i
                    break

            if i == len(preempt_list) - 1:
                break

            for i in range(st, len(preempt_list)):
                if preempt_list[i] == 1:
                    break
            ret.append((st, i - st))
            st = i
        return ret

    def get_idle_intervals(self, spot_used, preempt_list):
        # return a list of (idle_start_time, idle_life_time)
        ret = []
        st = 0
        spot_used_sorted = sorted(spot_used, key=lambda x: x[0])
        for spot_st, life in spot_used_sorted:
            if spot_st > st:
                ret.append((st, spot_st - st))
            st = spot_st + life
        if st < len(preempt_list):
            ret.append((st, len(preempt_list) - st))
        return ret

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)

        self.num_slices = self.args.num_slices

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
        self.slice_index = 0

        self.slice_gap_counts = [
            int(round(interval / env.gap_seconds, 0))
            for interval in self.slice_intervals
        ]
        self.slice_gap_counts[-1] = int(self.slice_intervals[-1] /
                                        env.gap_seconds)
        assert sum(self.slice_gap_counts) <= math.floor(
            self.deadline / env.gap_seconds)
        for i, slice_gap_counts in enumerate(self.slice_gap_counts[:-1]):
            assert abs(slice_gap_counts * env.gap_seconds -
                       self.slice_intervals[i]) < 1e-4, (
                           slice_gap_counts, env.gap_seconds,
                           self.slice_intervals[i])
            assert self.slice_task_durations[
                i] + self.restart_overhead <= self.slice_intervals[i]
        assert self.slice_task_durations[
            -1] + self.restart_overhead <= self.slice_gap_counts[
                -1] * env.gap_seconds

        last_slice_final_cluster_type = ClusterType.NONE
        self.slice_plans = []
        entire_trace = env.get_trace_before_end(self.deadline)
        for slice_idx in range(self.num_slices):
            slice_gap_start = sum(self.slice_gap_counts[:slice_idx])
            slice_gap_end = slice_gap_start + self.slice_gap_counts[slice_idx]
            trace = entire_trace[slice_gap_start:slice_gap_end]
            total_gaps = len(trace)
            task_duration = self.slice_task_durations[slice_idx]
            # Prepare constants
            on_demand_per_gap_cost = COSTS[
                ClusterType.ON_DEMAND] * env.gap_seconds / 3600
            spot_per_gap_cost = COSTS[
                ClusterType.SPOT] * env.gap_seconds / 3600
            # get a list of spot life time
            spot_lives = self.get_spot_intervals(trace)
            continue_spot_usage = (last_slice_final_cluster_type
                                   == ClusterType.SPOT
                                   and spot_lives[0][0] == 0)
            if continue_spot_usage:
                first_spot = spot_lives.pop(0)
            spot_lives_sorted = sorted(spot_lives,
                                       key=lambda x: x[1],
                                       reverse=True)
            if continue_spot_usage:
                spot_lives_sorted.insert(0, first_spot)

            cost_list = []
            search_done = False

            best_cost = 100000000
            best_spot_intervals = []
            best_vm_intervals = []
            for i in range(len(spot_lives_sorted) + 1):
                if search_done:
                    break

                cur_cost = 0
                # spot total hours
                used_spot_intervals = spot_lives_sorted[:i]
                spot_time = env.gap_seconds * sum(
                    [x[1] for x in spot_lives_sorted[:i]])
                required_task_duration = task_duration + self.restart_overhead * i
                if continue_spot_usage and i > 0:
                    required_task_duration -= self.restart_overhead
                if spot_time >= required_task_duration:
                    gap = spot_time - required_task_duration
                    spot_time = required_task_duration
                    last = used_spot_intervals.pop()
                    used_spot_intervals.append(
                        (last[0], last[1] - gap / env.gap_seconds))
                    search_done = True
                # minus overhead
                spot_progess = spot_time - self.restart_overhead * i
                if continue_spot_usage and i > 0:
                    spot_progess += self.restart_overhead

                cur_cost += spot_time / env.gap_seconds * spot_per_gap_cost

                progress_left = task_duration - spot_progess

                vm_time = 0
                used_vm_intervals = []
                if progress_left > 0:
                    # use on demand
                    idle_intervals = self.get_idle_intervals(
                        spot_lives_sorted[:i], trace)
                    continue_vm_usage = (last_slice_final_cluster_type
                                         == ClusterType.ON_DEMAND
                                         and idle_intervals[0][0] == 0)
                    if continue_vm_usage:
                        first_idle = idle_intervals.pop(0)
                    idle_intervals_sorted = sorted(idle_intervals,
                                                   key=lambda x: x[1],
                                                   reverse=True)
                    if continue_vm_usage:
                        idle_intervals_sorted.insert(0, first_idle)
                    for j in range(len(idle_intervals_sorted)):
                        restart_overhead = self.restart_overhead
                        if j == 0 and continue_vm_usage:
                            restart_overhead = 0
                        if env.gap_seconds * idle_intervals_sorted[j][
                                1] >= progress_left + restart_overhead:
                            used_vm_intervals.append(
                                (idle_intervals_sorted[j][0],
                                 (progress_left + restart_overhead) /
                                 env.gap_seconds))
                            vm_time += progress_left + restart_overhead
                            progress_left = 0
                            break
                        vm_time += env.gap_seconds * idle_intervals_sorted[j][1]
                        used_vm_intervals.append(idle_intervals_sorted[j])
                        progress_left -= (
                            env.gap_seconds * idle_intervals_sorted[j][1] -
                            restart_overhead)

                if progress_left == 0:
                    cur_cost += vm_time / env.gap_seconds * on_demand_per_gap_cost
                    if cur_cost < best_cost:
                        best_spot_intervals = used_spot_intervals
                        best_vm_intervals = used_vm_intervals
                        best_cost = cur_cost
                    cost_list.append(cur_cost)

            plan = [ClusterType.NONE] * total_gaps
            for spot_st, life in best_spot_intervals:
                for i in range(spot_st, spot_st + math.ceil(life)):
                    plan[i] = ClusterType.SPOT
            for vm_st, life in best_vm_intervals:
                for i in range(vm_st, vm_st + math.ceil(life)):
                    plan[i] = ClusterType.ON_DEMAND
            self.slice_plans.append(plan)
            last_slice_final_cluster_type = plan[-1]

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        if self.slice_index >= self.num_slices:
            return ClusterType.NONE
        slice_end_gap_index = sum(self.slice_gap_counts[:self.slice_index + 1])
        if slice_end_gap_index <= len(self.task_done_time) - 1:
            assert self.env.tick - sum(
                self.slice_gap_counts[:self.slice_index]) == len(
                    self.slice_plans[self.slice_index])
            self.slice_index += 1
            if self.slice_index >= self.num_slices:
                delta = self.task_duration - sum(self.task_done_time)
                assert delta <= self.env.gap_seconds, delta
                return ClusterType.NONE
        plan = self.slice_plans[self.slice_index]
        tick_in_slice = self.env.tick - sum(
            self.slice_gap_counts[:self.slice_index])
        request_cluster_type = plan[tick_in_slice]
        assert request_cluster_type != ClusterType.SPOT or has_spot, (
            self.env.tick, self.slice_index, plan, tick_in_slice)
        return request_cluster_type

    @classmethod
    def _from_args(
        cls, parser: 'argparse.ArgumentParser'
    ) -> 'QuickOptimalSlicedByNumStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        group.add_argument('--num-slices', type=int, default=1)
        args, _ = parser.parse_known_args()
        return cls(args)
