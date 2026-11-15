from dataclasses import dataclass
from typing import Optional, Union
from sky_spot.utils import ClusterType


@dataclass
class TryLaunch:
    region: int
    cluster_type: ClusterType


@dataclass
class Terminate:
    region: int


@dataclass
class LaunchResult:
    success: bool
    region: int
    cluster_type: Optional[ClusterType] = None


# Type alias for all possible actions
Action = Union[TryLaunch, Terminate] 