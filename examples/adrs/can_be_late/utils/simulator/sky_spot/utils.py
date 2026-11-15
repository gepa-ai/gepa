import enum


class ClusterType(str, enum.Enum):
    NONE = enum.auto()
    SPOT = enum.auto()
    ON_DEMAND = enum.auto()


# Price for p3.2xlarge (single V100) on us-west-2
# https://aws.amazon.com/ec2/instance-types/p3/
COSTS = {
    ClusterType.ON_DEMAND: 3.06,
    ClusterType.SPOT: 0.9731,
    ClusterType.NONE: 0,
}
COST_K = COSTS[ClusterType.ON_DEMAND] / COSTS[ClusterType.SPOT]

COST_SCALES = {
    'v100_1': 1,
    'v100_8': 24.48 / COSTS[ClusterType.ON_DEMAND],
    'k80_1': 0.9 / COSTS[ClusterType.ON_DEMAND],
    'k80_8': 14.400 / COSTS[ClusterType.ON_DEMAND],
}


DEVICE_COSTS = {
    'v100_1': 3.06,
    'v100_8': 3.06, # this is wrong but to be consistent with the original results 24.48,
    'k80_1': 3.06, # this is wrong but to be consistent with the original results 0.9,
    'k80_8': 3.06, # this is wrong but to be consistent with the original results 14.400,
    't4_8': 7.82,
    't4_4': 3.91,
    'intel_64': 4.03,
    'intel_48': 3.02,
    'a10g_4': 5.67,
    'c2-60': 3.1321,
    'c3-88': 6.20048,
    'c3-176': 12.40096,
}

ACTUAL_COSTS = {
    'v100_1': (3.06, 0.918),
    'intel_64': (4.03, 1.8515),
    'c3-88': (3.79, 0.35),
    'c3-176': (7.59, 0.69)
}
# Price for p2.xlarge (single K80) on us-east-1
# COSTS = {
#     ClusterType.ON_DEMAND: 0.9,
#     ClusterType.SPOT: 0.3384,
#     ClusterType.NONE: 0,
# }


def wandb_log(*args, **kwargs):
    import wandb
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
