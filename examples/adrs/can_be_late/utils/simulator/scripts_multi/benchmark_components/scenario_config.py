"""Scenario configuration module - self-contained."""

# Standard strategy set for all scenarios
STANDARD_STRATEGIES = [
    "multi_region_rc_cr_threshold",
    "multi_region_rc_cr_no_cond2",
    "multi_region_rc_cr_randomized",
    "multi_region_rc_cr_reactive",
    "lazy_cost_aware_multi",
    "evolutionary_simple_v2",
    "quick_optimal",
    "best_110",
    "best_210",
]

US_EAST_1_REGIONS = [
    "us-east-1a_v100_1",
    "us-east-1c_v100_1", 
    "us-east-1d_v100_1",
    "us-east-1f_v100_1"
]

US_EAST_2_REGIONS = [
    "us-east-2a_v100_1",
    "us-east-2b_v100_1"
]

US_WEST_2_REGIONS = [
    "us-west-2a_v100_1",
    "us-west-2b_v100_1",
    "us-west-2c_v100_1"
]


# Systematic experiment scenarios organized by geographic regions
EXPERIMENT_SCENARIOS = [
    {
        "name": "us-east-1 Complete (4 Zones)",
        "regions": US_EAST_1_REGIONS,
        "strategies": STANDARD_STRATEGIES,
        "compare_single_region": True,
        "description": "Complete us-east-1 region with all availability zones"
    },
    {
        "name": "us-east-2 Complete (2 Zones)", 
        "regions": US_EAST_2_REGIONS,
        "strategies": STANDARD_STRATEGIES,
        "compare_single_region": True,
        "description": "Complete us-east-2 region with all availability zones"
    },
    {
        "name": "us-west-2 Complete (3 Zones)",
        "regions": US_WEST_2_REGIONS,
        "strategies": STANDARD_STRATEGIES,
        "compare_single_region": True,
        "description": "Complete us-west-2 region with all availability zones"
    },
    {
        "name": "East Coast Complete (6 Regions)",
        "regions": US_EAST_1_REGIONS + US_EAST_2_REGIONS,
        "strategies": STANDARD_STRATEGIES,
        "compare_single_region": True,
        "description": "All East Coast regions (us-east-1 + us-east-2)"
    },
    {
        "name": "Cross-Coast (East + West)",
        "regions": US_EAST_1_REGIONS + US_EAST_2_REGIONS + US_WEST_2_REGIONS,
        "strategies": STANDARD_STRATEGIES,
        "compare_single_region": True,
        "description": "Representative cross-coast scenario for latency testing"
    },
]

# Default benchmark parameters
DEFAULT_PARAMS = {
    "TASK_DURATION_HOURS": 44.0,
    "DEADLINE_HOURS": 60.0,
    "ENV_START_HOURS": 0.0,
    "MAX_WORKERS": 4,
    "DATA_PATH": "data/converted_multi_region_aligned",
    "OUTPUT_DIR": "outputs/multi_region_scenario_analysis",
    "SINGLE_REGION_STRATEGY": "rc_cr_threshold"
}