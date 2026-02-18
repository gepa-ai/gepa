import argparse
import typing

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env
    from sky_spot import task


class IdealNoOverheadStrategy(strategy.Strategy):
    NAME = 'ideal_no_overhead'

    def reset(self, env: 'env.Env', task: 'task.Task'):
        super().reset(env, task)

        self.trace = env.get_trace_before_end(self.deadline)

    def _step(self, last_cluster_type: ClusterType,
              has_spot: bool) -> ClusterType:
        if self.task_done:
            return ClusterType.NONE

        if has_spot:
            return ClusterType.SPOT

        remaining_time = self.deadline - self.env.elapsed_seconds
        task_remaining_time = self.task_duration - sum(self.task_done_time)
        if task_remaining_time >= remaining_time:
            return ClusterType.ON_DEMAND

        return ClusterType.NONE

    @classmethod
    def _from_args(
            cls,
            parser: 'argparse.ArgumentParser') -> 'IdealNoOverheadStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
