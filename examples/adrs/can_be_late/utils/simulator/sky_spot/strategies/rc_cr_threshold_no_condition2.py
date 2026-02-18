"""
Our policy: Uniform Progress.
"""

import argparse
import json
import logging
import math

from sky_spot.strategies import rc_threshold
from sky_spot.utils import ClusterType

import typing

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task

logger = logging.getLogger(__name__)


class RCCRThresholdStrategy(rc_threshold.RCThresholdStrategy):
    NAME = "rc_cr_threshold_no_condition2"

    def __init__(self, args):
        args.keep_on_demand = None
        super().__init__(args)

    def reset(self, env: "env.Env", task: "task.Task"):
        super().reset(env, task)
        self.cur_on_demand_time = 0

    def _condition(self):
        c_0 = self.task_duration
        c_t = self.task_duration - sum(self.task_done_time)
        t = self.env.elapsed_seconds
        r_0 = self.deadline
        return c_0 - c_t - t * c_0 / r_0

    def _condition2(self):
        # When on VM, only switch back to spot only when the condition2 < 0
        d = self.restart_overhead
        t = self.env.elapsed_seconds
        c_t = self.task_duration - sum(self.task_done_time)
        c_0 = self.task_duration
        r_0 = self.deadline
        return (t + 2 * d) * c_0 / r_0 - (c_0 - c_t)

    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        env = self.env

        # Make decision for the gap starting from env.tick
        remaining_time = (
            math.floor((self.deadline - env.elapsed_seconds) / env.gap_seconds)
            * env.gap_seconds
        )
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
            if not has_spot:
                # Keep the on-demand VM until the end
                logger.debug(f"{env.tick}: Keep on-demand VM")
                request_type = ClusterType.ON_DEMAND

        total_task_remaining = (
            math.ceil(
                (remaining_task_time + self.restart_overhead) / self.env.gap_seconds
            )
            * self.env.gap_seconds
        )

        total_task_remaining_with_2D = (
            math.ceil(
                (remaining_task_time + 2 * self.restart_overhead) / self.env.gap_seconds
            )
            * self.env.gap_seconds
        )
        if total_task_remaining_with_2D >= remaining_time and current_cluster_type in [
            ClusterType.NONE,
            ClusterType.ON_DEMAND,
        ]:
            # We need to switch to on-demand when it is T+2D
            logger.debug(f"{env.tick}: Deadline reached, switch to on-demand")
            request_type = ClusterType.ON_DEMAND

        if total_task_remaining >= remaining_time:
            if (
                current_cluster_type == ClusterType.SPOT
                and self.remaining_restart_overhead < 1e-3
            ):
                # Keep the spot VM until preemption
                logger.debug(
                    f"{env.tick}: Deadline reached, keep spot until preemption"
                )
                request_type = ClusterType.SPOT
            else:
                logger.debug(f"{env.tick}: Deadline reached, switch to on-demand")
                # We need to finish it on time by switch to on-demand
                request_type = ClusterType.ON_DEMAND
            if self.restart_overhead == 0 and has_spot:
                # We can switch to spot without cost.
                request_type = ClusterType.SPOT

        if request_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time += self.env.gap_seconds
        elif last_cluster_type == ClusterType.ON_DEMAND:
            self.cur_on_demand_time = 0
        log_data = {
            "tick": env.tick,
            "decision": request_type.name,
            # For single region, region is always 0 and there's no "global" view
            "current_region": 0,
            "global_has_spot": has_spot,
            "best_spot_region": 0 if has_spot else None,
            "accumulated_cost": env.accumulated_cost,
            "remaining_task_time": remaining_task_time,
            "remaining_overhead": self.remaining_restart_overhead,
            "last_tick_work": self.task_done_time[-1] if self.task_done_time else 0,
        }
        logger.info(f"STRATEGY_DECISION: {json.dumps(log_data)}")

        return request_type

    @classmethod
    def _from_args(cls, parser: "argparse.ArgumentParser") -> "RCCRThresholdStrategy":
        group = parser.add_argument_group("RCV2DThresholdStrategy")
        args, _ = parser.parse_known_args()
        return cls(args)
