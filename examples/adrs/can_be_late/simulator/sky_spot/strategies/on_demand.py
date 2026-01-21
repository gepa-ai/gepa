import argparse
import typing

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env


class OnDemandStrategy(strategy.Strategy):
    NAME = 'on_demand'

    def _step(self, last_cluster_type, has_spot) -> ClusterType:
        if self.task_done:
            return ClusterType.NONE

        # Make decision for the gap starting from env.tick
        return ClusterType.ON_DEMAND

    @classmethod
    def _from_args(cls,
                   parser: 'argparse.ArgumentParser') -> 'OnDemandStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
