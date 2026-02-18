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


class QuickOptimalParanoidStrategy(strategy.Strategy):
    NAME = 'quick_optimal_paranoid'

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

    def get_idle_intervals(self, spot_used):
        # return a list of (idle_start_time, idle_life_time)
        ret = []
        st = 0
        spot_used_sorted = sorted(spot_used, key=lambda x: x[0])
        for spot_st, life in spot_used_sorted:
            if spot_st > st:
                ret.append((st, spot_st - st))
            st = spot_st + life
        if st < len(self.trace):
            ret.append((st, len(self.trace) - st))
        return ret

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)

        self.trace = env.get_trace_before_end(self.deadline)
        self.total_gaps = math.ceil(self.deadline / env.gap_seconds)

        # Prepare constants
        on_demand_per_gap_cost = COSTS[
            ClusterType.ON_DEMAND] * env.gap_seconds / 3600
        spot_per_gap_cost = COSTS[ClusterType.SPOT] * env.gap_seconds / 3600

        # get a list of spot life time
        spot_lives = self.get_spot_intervals(self.trace)
        spot_lives_sorted = sorted(spot_lives,
                                   key=lambda x: x[1],
                                   reverse=True)

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
            if spot_time >= self.task_duration + self.restart_overhead * i:
                gap = spot_time - self.task_duration - self.restart_overhead * i
                spot_time = self.task_duration + self.restart_overhead * i
                last = used_spot_intervals.pop()
                used_spot_intervals.append(
                    (last[0], last[1] - gap / env.gap_seconds))
                search_done = True
            # minus overhead
            spot_progess = spot_time - self.restart_overhead * i

            cur_cost += spot_time / env.gap_seconds * spot_per_gap_cost

            progress_left = self.task_duration - spot_progess

            vm_time = 0
            used_vm_intervals = []
            if progress_left > 0:
                # use on demand
                idle_intervals = self.get_idle_intervals(spot_lives_sorted[:i])
                idle_intervals_sorted = sorted(idle_intervals,
                                               key=lambda x: x[1],
                                               reverse=True)
                for j in range(len(idle_intervals_sorted)):
                    if env.gap_seconds * idle_intervals_sorted[j][
                            1] >= progress_left + self.restart_overhead:
                        used_vm_intervals.append(
                            (idle_intervals_sorted[j][0],
                             (progress_left + self.restart_overhead) /
                             env.gap_seconds))
                        vm_time += progress_left + self.restart_overhead
                        progress_left = 0
                        break
                    vm_time += env.gap_seconds * idle_intervals_sorted[j][1]
                    used_vm_intervals.append(idle_intervals_sorted[j])
                    progress_left -= (
                        env.gap_seconds * idle_intervals_sorted[j][1] -
                        self.restart_overhead)

            if progress_left == 0:
                cur_cost += vm_time / env.gap_seconds * on_demand_per_gap_cost
                if cur_cost < best_cost:
                    best_spot_intervals = used_spot_intervals
                    best_vm_intervals = used_vm_intervals
                    best_cost = cur_cost
                cost_list.append(cur_cost)

        self.plan = [ClusterType.NONE] * self.total_gaps
        for spot_st, life in best_spot_intervals:
            for i in range(spot_st, spot_st + math.ceil(life)):
                self.plan[i] = ClusterType.SPOT
        for vm_st, life in best_vm_intervals:
            for i in range(vm_st, vm_st + math.ceil(life)):
                self.plan[i] = ClusterType.ON_DEMAND

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        if self.env.tick >= self.total_gaps:
            return ClusterType.NONE
        request_cluster_type = self.plan[self.env.tick]

        remaining_time = math.floor(
            (self.deadline - self.env.elapsed_seconds) /
            self.env.gap_seconds) * self.env.gap_seconds
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        if remaining_task_time <= 1e-3:
            return ClusterType.NONE
        total_task_remaining = math.ceil(
            (remaining_task_time + self.restart_overhead) /
            self.env.gap_seconds) * self.env.gap_seconds

        current_cluster_type = self.env.cluster_type
        if total_task_remaining >= remaining_time:
            if current_cluster_type == ClusterType.SPOT and self.remaining_restart_overhead < 1e-3:
                # Keep the spot VM until preemption
                request_cluster_type = ClusterType.SPOT
            else:
                # We need to finish it on time by switch to on-demand
                request_cluster_type = ClusterType.ON_DEMAND

        assert request_cluster_type != ClusterType.SPOT or has_spot
        return request_cluster_type

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser'
    ) -> 'QuickOptimalParanoidStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
