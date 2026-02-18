import argparse
import math
import multiprocessing
import os
import time
import typing

import pulp

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


class IdealILPStrategy(strategy.Strategy):
    NAME = 'ideal_ilp_overhead'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)

        self.trace = env.get_trace_before_end(self.deadline)
        self.total_gaps = math.floor(self.deadline / env.gap_seconds)

        # Solve ILP for ideal strategy
        tic = time.time()
        prob = pulp.LpProblem("Cost-Minimization", pulp.LpMinimize)

        # Prepare constants
        cost_map = env.get_constant_cost_map()
        on_demand_per_gap_cost = cost_map[
            ClusterType.ON_DEMAND] * env.gap_seconds / 3600
        spot_per_gap_cost = cost_map[ClusterType.SPOT] * env.gap_seconds / 3600

        # Define the variables
        a = pulp.LpVariable.dicts("a",
                                  range(self.total_gaps),
                                  cat=pulp.LpBinary)
        b = pulp.LpVariable.dicts("b",
                                  range(self.total_gaps),
                                  cat=pulp.LpBinary)
        x = pulp.LpVariable.dicts("x",
                                  range(self.total_gaps - 1),
                                  cat=pulp.LpBinary)  # intermediate variable
        y = pulp.LpVariable.dicts("y",
                                  range(self.total_gaps - 1),
                                  cat=pulp.LpBinary)  # intermediate variable

        # Define the objective function
        objective = pulp.lpSum(a[i] * spot_per_gap_cost +
                               b[i] * on_demand_per_gap_cost
                               for i in range(self.total_gaps))
        prob += objective, "Total Cost"

        # Define the constraints
        for i in range(self.total_gaps):
            prob += a[i] + b[
                i] <= 1, f"Only one of the a[{i}], b[{i}] variables can be 1"
            prob += a[i] <= (
                1 - self.trace[i]), f"a[{i}] is 1 only if there is spot"
            if i < self.total_gaps - 1:
                prob += x[i] <= a[i +
                                  1], f"x[{i}] can be 1 only if a[{i+1}] is 1"
                prob += x[i] <= 1 - a[i], f"x[{i}] can be 1 only if a[{i}] is 0"
                prob += x[i] >= a[i + 1] - a[
                    i], f"x[{i}] can be 1 only if a[{i+1}] is 1 and a[{i}] is 0"
                prob += y[i] <= b[i +
                                  1], f"y[{i}] can be 1 only if b[{i+1}] is 1"
                prob += y[i] <= 1 - b[i], f"y[{i}] can be 1 only if b[{i}] is 0"
                prob += y[i] >= b[i + 1] - b[
                    i], f"y[{i}] can be 1 only if b[{i+1}] is 1 and b[{i}] is 0"
        H = self.restart_overhead * (
            pulp.lpSum(x[i] + y[i]
                       for i in range(self.total_gaps - 1)) + a[0] + b[0]
        )  # a[0] + b[0] for the first gap
        prob += env.gap_seconds * pulp.lpSum(a[i] + b[i] for i in range(
            self.total_gaps)) >= H + self.task_duration, "Task duration"

        verbose = False

        msg = verbose
        time_limit = 600
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
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if verbose:
            print(
                f"ILP Status: {pulp.LpStatus[status]}\tObjective: {objective}\t"
                f"Time: {time.time() - tic}")
        self.plan = []
        for i in range(self.total_gaps):
            if pulp.value(a[i]) == 1:
                self.plan.append(ClusterType.SPOT)
            elif pulp.value(b[i]) == 1:
                self.plan.append(ClusterType.ON_DEMAND)
            else:
                self.plan.append(ClusterType.NONE)

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        if self.env.tick >= self.total_gaps:
            return ClusterType.NONE
        request_cluster_type = self.plan[self.env.tick]
        assert request_cluster_type != ClusterType.SPOT or has_spot
        return request_cluster_type

    @classmethod
    def _from_args(cls,
                   parser: 'argparse.ArgumentParser') -> 'IdealILPStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
