import argparse
import typing

from sky_spot.strategies import strategy
from sky_spot.utils import ClusterType

if typing.TYPE_CHECKING:
    from sky_spot import env


class OnlySpotStrategy(strategy.Strategy):
    NAME = 'only_spot'

    def _step(self, last_cluster_type, has_spot) -> ClusterType:
        if self.task_done:
            return ClusterType.NONE

        if has_spot:
            return ClusterType.SPOT
        else:
            return ClusterType.NONE

    @classmethod
    def _from_args(cls,
                   parser: 'argparse.ArgumentParser') -> 'OnlySpotStrategy':
        group = parser.add_argument_group('OnDemandStrategy')
        args, _ = parser.parse_known_args()
        return cls(args)
