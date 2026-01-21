import argparse
import math
import multiprocessing
import os
import time
import typing

import pulp

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType, COSTS

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


class IdealILPSlicedByNumStrategy(strategy.Strategy):
    NAME = 'ideal_ilp_overhead_sliced_by_num'

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
        self.slice_gap_counts = None
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

        # Calculate the optimal strategy within each slice
        last_slice_final_cluster_type = ClusterType.NONE
        self.slice_plans = []
        entire_trace = env.get_trace_before_end(self.deadline)
        cost_map = env.get_constant_cost_map()
        for slice_idx in range(self.num_slices):
            slice_gap_start = sum(self.slice_gap_counts[:slice_idx])
            slice_gap_end = slice_gap_start + self.slice_gap_counts[slice_idx]
            trace = entire_trace[slice_gap_start:slice_gap_end]
            total_gaps = len(trace)
            assert total_gaps == self.slice_gap_counts[slice_idx]

            # Solve ILP for ideal strategy
            tic = time.time()
            prob = pulp.LpProblem("Cost-Minimization", pulp.LpMinimize)

            # Prepare constants
            on_demand_per_gap_cost = cost_map[
                ClusterType.ON_DEMAND] * env.gap_seconds / 3600
            spot_per_gap_cost = cost_map[
                ClusterType.SPOT] * env.gap_seconds / 3600

            # Define the variables
            a = pulp.LpVariable.dicts("a",
                                      range(total_gaps),
                                      cat=pulp.LpBinary)
            b = pulp.LpVariable.dicts("b",
                                      range(total_gaps),
                                      cat=pulp.LpBinary)
            x = pulp.LpVariable.dicts(
                "x", range(total_gaps - 1),
                cat=pulp.LpBinary)  # intermediate variable
            y = pulp.LpVariable.dicts(
                "y", range(total_gaps - 1),
                cat=pulp.LpBinary)  # intermediate variable

            # Define the objective function
            objective = pulp.lpSum(a[i] * spot_per_gap_cost +
                                   b[i] * on_demand_per_gap_cost
                                   for i in range(total_gaps))
            prob += objective, "Total Cost"

            # Define the constraints
            for i in range(total_gaps):
                prob += a[i] + b[
                    i] <= 1, f"Only one of the a[{i}], b[{i}] variables can be 1"
                prob += a[i] <= (
                    1 - trace[i]), f"a[{i}] is 1 only if there is spot"
                if i < total_gaps - 1:
                    prob += x[i] <= a[
                        i + 1], f"x[{i}] can be 1 only if a[{i+1}] is 1"
                    prob += x[
                        i] <= 1 - a[i], f"x[{i}] can be 1 only if a[{i}] is 0"
                    prob += x[i] >= a[i + 1] - a[
                        i], f"x[{i}] can be 1 only if a[{i+1}] is 1 and a[{i}] is 0"
                    prob += y[i] <= b[
                        i + 1], f"y[{i}] can be 1 only if b[{i+1}] is 1"
                    prob += y[
                        i] <= 1 - b[i], f"y[{i}] can be 1 only if b[{i}] is 0"
                    prob += y[i] >= b[i + 1] - b[
                        i], f"y[{i}] can be 1 only if b[{i+1}] is 1 and b[{i}] is 0"

            # +1 for the first gap if it fails to match the last slice
            if last_slice_final_cluster_type == ClusterType.ON_DEMAND:
                first_gap_overhead = a[0]
            elif last_slice_final_cluster_type == ClusterType.SPOT:
                first_gap_overhead = b[0]
            else:
                first_gap_overhead = a[0] + b[0]

            H = self.restart_overhead * (pulp.lpSum(
                [x[i] + y[i]
                 for i in range(total_gaps - 1)] + [first_gap_overhead]))

            prob += env.gap_seconds * pulp.lpSum(
                a[i] + b[i] for i in range(total_gaps)
            ) >= H + self.slice_task_durations[slice_idx], "Task duration"

            verbose = False

            msg = verbose
            time_limit = 1200
            assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
                "Please install ILP solvers by 'sudo apt install coinor-cbc'")

            solver = pulp.PULP_CBC_CMD(mip=True,
                                       msg=msg,
                                       timeLimit=time_limit,
                                       threads=multiprocessing.cpu_count())

            tmpdir = os.path.expanduser('~/solver_tmp')
            os.makedirs(tmpdir, exist_ok=True)
            solver.tmpDir = tmpdir
            prob.solve(solver)
            status = prob.status
            assert status == pulp.LpStatusOptimal, (
                status, self, slice_idx, len(trace),
                self.slice_task_durations[slice_idx])
            objective = pulp.value(prob.objective)
            objective = float(objective) if objective is not None else -1.0
            if verbose:
                print(
                    f"ILP Status: {pulp.LpStatus[status]}\tObjective: {objective}\t"
                    f"Time: {time.time() - tic}")
            plan = []
            cur_type = last_slice_final_cluster_type
            task_done = 0
            a_list = []
            b_list = []
            for i in range(total_gaps):
                a_list.append(pulp.value(a[i]))
                b_list.append(pulp.value(b[i]))
                if pulp.value(a[i]) == 1:
                    plan.append(ClusterType.SPOT)
                    assert 1 - trace[i] == 1, (i, trace[i])
                elif pulp.value(b[i]) == 1:
                    plan.append(ClusterType.ON_DEMAND)
                else:
                    plan.append(ClusterType.NONE)

                if plan[-1] != ClusterType.NONE:
                    if cur_type != plan[-1]:
                        task_done -= self.restart_overhead
                    task_done += self.env.gap_seconds
                    cur_type = plan[-1]
            assert task_done >= self.slice_task_durations[slice_idx], (
                task_done, self.slice_task_durations[slice_idx],
                self.slice_index, a_list, b_list, plan)
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

    @property
    def config(self):
        new_config = super().config
        new_config.update({
            'num_slices': self.num_slices,
        })
        return new_config

    @classmethod
    def _from_args(
            cls, parser: 'argparse.ArgumentParser'
    ) -> 'IdealILPSlicedByNumStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        group.add_argument('--num-slices', type=int, default=1)
        args, _ = parser.parse_known_args()
        return cls(args)
