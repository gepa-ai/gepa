import json
import logging
import os
import typing
from typing import Sequence, Optional

import tqdm
import wandb
import numpy as np

from sky_spot import env as env_lib
# from sky_spot.env import SubtaskMultiEnvSwitcher
from sky_spot.strategies import strategy as strategy_lib
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.utils import ClusterType, wandb_log
from sky_spot import task as task_lib

logger = logging.getLogger(__name__)


def _simulate_one(env: env_lib.Env, strategy: strategy_lib.Strategy):
    history = []
    last_request_type = ClusterType.NONE
    
    # Check if this is a multi-region setup
    is_multi_region = (
        isinstance(strategy, MultiRegionStrategy) and 
        isinstance(env, env_lib.MultiTraceEnv)
    )
    multi_env = typing.cast(env_lib.MultiTraceEnv, env)
    multi_strategy = typing.cast(MultiRegionStrategy, strategy)

    while not strategy.task_done:
        if is_multi_region:
            # Multi-region execution path
            multi_env.observe()
            multi_env.update_strategy_progress(multi_strategy)  # Update task progress based on PREVIOUS tick
            
            # Check if task became done after progress update
            if strategy.task_done:
                break
                
            multi_env.execute_multi_strategy(multi_strategy)
            multi_env.tick += 1
            
            # Finalize costs for this tick (they will be recorded in next observe())
            # This ensures costs are properly tracked even for the last tick
            
            # For history tracking, use a representative request type
            # (could be enhanced to track all regions)
            active = multi_env.get_active_instances()
            if active:
                # Use the type of the first active instance as representative
                request_type = next(iter(active.values()))
            else:
                request_type = ClusterType.NONE
        else:
            # Single-region execution path (original)
            request_type = strategy.step()
            env.step(request_type)

        info = {
            "RequestType": last_request_type.value if not is_multi_region else request_type.value,
            **env.info(),
            **strategy.info(),
        }
        
        # Add multi-region specific info if applicable
        if is_multi_region:
            cost_breakdown = multi_env.get_cost_breakdown()
            # Convert ClusterType keys to strings for JSON serialization
            cost_by_type_str = {k.name: v for k, v in cost_breakdown['by_type'].items()}
            
            # Add spot availability for each region
            spot_availability = {}
            for region in range(multi_env.num_regions):
                spot_availability[region] = multi_env._spot_available_in_region(region)
                    
            info.update({
                "ActiveRegions": len(multi_env.get_active_instances()),
                "CostByRegion": cost_breakdown['by_region'],
                "CostByType": cost_by_type_str,
                "SpotAvailability": spot_availability,
                "ActiveInstances": {k: v.name for k, v in multi_env.get_active_instances().items()},
                "MigrationCount": multi_env.migration_count,
            })
        
        last_request_type = request_type
        history.append(info)
        wandb_log(info)
        if env.tick % 100 == 0:
            logger.debug(f"==> Timestamp: {env.tick}")

    # Final step after task is done
    if is_multi_region:
        # The main loop may have exited with unfinalzied costs from the last execute
        # We need to finalize them before getting the final cost
        multi_env._finalize_tick_costs()
        # Now call observe to update observed_tick
        multi_env.tick += 1
        multi_env.observed_tick = multi_env.tick - 1
    else:
        strategy.step()  # realize the last step
        env.step(ClusterType.NONE)
    
    info = {
        "RequestType": ClusterType.NONE.value,
        **env.info(),
        **strategy.info(),
    }
    
    if is_multi_region:
        cost_breakdown = multi_env.get_cost_breakdown()
        info.update({
            "ActiveRegions": 0,
            "CostByRegion": cost_breakdown['by_region'],
            "CostByType": cost_breakdown['by_type'],
        })
    
    return history, env.tick


def simulate(
    envs: Sequence[env_lib.Env],
    strategy: strategy_lib.Strategy,
    task: task_lib.Task,
    trace_file: str,
    deadline_hours: float,
    restart_overhead_hours: list[float],
    env_start_hours: float,
    output_dir: str,
    kwargs: dict,
    output_filename: Optional[str] = None,
    silent: bool = False,
    dump_history: bool = True,
):
    # Pre-check: Ensure task is feasible within deadline
    task_duration_hours = task.get_total_duration_hours()
    assert len(restart_overhead_hours) == 1, "Only one restart overhead is supported"
    max_restart_overhead = max(restart_overhead_hours) if restart_overhead_hours else 0
    min_time_needed = task_duration_hours + max_restart_overhead
    
    if min_time_needed > deadline_hours:
        error_msg = (
            f"Task infeasible: minimum time needed ({min_time_needed:.2f}h = "
            f"task {task_duration_hours:.2f}h + max overhead {max_restart_overhead:.2f}h) "
            f"exceeds deadline ({deadline_hours:.2f}h)"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Pre-check: Ensure trace data covers the deadline period
    # This check is important to prevent out-of-bounds errors during simulation
    for i, env in enumerate(envs):
        if hasattr(env, 'trace') and hasattr(env, '_start_index'):
            # Single-region environment
            trace_duration_seconds = (len(env.trace) - env._start_index) * env.gap_seconds
            trace_duration_hours = trace_duration_seconds / 3600
            
            if trace_duration_hours < deadline_hours:
                error_msg = (
                    f"Trace data insufficient: trace {i} duration ({trace_duration_hours:.2f}h) "
                    f"is less than deadline ({deadline_hours:.2f}h). "
                    f"Trace has {len(env.trace)} ticks with gap {env.gap_seconds}s, "
                    f"starting from index {env._start_index}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif hasattr(env, 'envs'):
            # Multi-region environment
            for region_idx, region_env in enumerate(env.envs):
                if hasattr(region_env, 'trace') and hasattr(region_env, '_start_index'):
                    trace_duration_seconds = (len(region_env.trace) - region_env._start_index) * region_env.gap_seconds
                    trace_duration_hours = trace_duration_seconds / 3600
                    
                    if trace_duration_hours < deadline_hours:
                        error_msg = (
                            f"Trace data insufficient: region {region_idx} trace duration ({trace_duration_hours:.2f}h) "
                            f"is less than deadline ({deadline_hours:.2f}h). "
                            f"Trace has {len(region_env.trace)} ticks with gap {region_env.gap_seconds}s, "
                            f"starting from index {region_env._start_index}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
    
    histories = []
    costs = []
    ticks = []
    migrations = []

    trace_file = trace_file.split("/")[-1]
    env_name = envs[0].NAME
    env_config = envs[0].config
    # RESETTING STRATEGY IS VERY IMPORTANT!
    strategy.reset(envs[0], task)

    restart_overhead_str = "_".join(map(str, restart_overhead_hours))
    run_name = f"{strategy.name}-{env_name}-{trace_file}-ddl={deadline_hours}-task={task}-over={restart_overhead_str}"
    if env_start_hours > 0:
        run_name += f"-start={env_start_hours}h"
    logger.debug(run_name)
    if not silent and wandb.run is not None:
        wandb.run.name = run_name
        wandb.config.update(
            {
                "trace_file": trace_file,
                "deadline_hours": deadline_hours,
                "restart_overhead_hours": restart_overhead_hours,
                "env_start_hours": env_start_hours,
                "task_config": task.get_config(),
                "other_args": {
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "deadline_hours",
                        "task_duration_hours",
                        "task_duration_hours_2",
                        "restart_overhead_hours",
                        "env_start_hours",
                    ]
                },
            }
        )
        wandb.config.update({"env_metadata": env_config})
        wandb.config.update({"strategy_metadata": strategy.config})

    if silent:
        pbar = envs
    else:
        pbar = tqdm.tqdm(envs)
    for env in pbar:
        # pbar.set_description(f'env: {env}')
        # ! Must reset env and strategy at the first time!
        # ! reset is their init method
        env.reset()
        strategy.reset(env, task)

        #         if isinstance(env, SubtaskMultiEnvSwitcher):
        #             if isinstance(task, task_lib.ChainedTask):
        #                 env.set_task(task)
        #                 logger.debug("Associated ChainedTask with SubtaskMultiEnvSwitcher.")
        #             else:
        #                 raise ValueError(
        #                     "SubtaskMultiEnvSwitcher requires a ChainedTask, but received a different task type."
        #                 )

        logger.debug(kwargs)
        logger.debug(env)
        logger.debug(strategy)

        history, tick = _simulate_one(env, strategy)
        histories.append(history)
        costs.append(history[-1]["Cost"])
        ticks.append(tick)
        
        # Extract migration count if available (multi-region)
        migration_count = history[-1].get("MigrationCount", 0)
        migrations.append(migration_count)

        # if len(envs) > 1:
        #     env.reset()
        #     new_args = copy.deepcopy(args)
        #     new_args.deadline_hours = 1000
        #     spot_strategy = sky_spot.strategies.only_spot.OnlySpotStrategy(new_args)
        #     spot_costs.append(simulate(env, spot_strategy))
        #     cost_ratio.append(costs[-1] / spot_costs[-1])

        # mean_strategy_cost = np.mean(costs)
        # std_strategy_cost = np.std(costs)
        # mean_spot_cost = np.mean(spot_costs)
        # std_spot_cost = np.std(spot_costs)
        # mean_cost_ratio = np.mean(cost_ratio)
        # std_cost_ratio = np.std(cost_ratio)
        # msg = f'cost: {mean_strategy_cost:.2f}±{std_strategy_cost:.2f}; spot cost: {mean_spot_cost:.2f}±{std_spot_cost:.2f}; cost ratio: {mean_cost_ratio:.2f}±{std_cost_ratio:.2f}'
        # logger.debug('=== ' + msg + ' ===')
        # pbar.set_description(msg)
        # wandb.log({'MeanCost': mean_strategy_cost, 'StdCost': std_strategy_cost, 'MeanSpotCost': mean_spot_cost, 'StdSpotCost': std_spot_cost, 'MeanCostRatio': mean_cost_ratio, 'StdCostRatio': std_cost_ratio})

    os.makedirs(output_dir, exist_ok=True)
    if output_filename is not None:
        run_name = output_filename
    stats = {
        "args": kwargs,
        "costs": costs,
        "migrations": migrations,
        "strategy": strategy.config,
        "env": env_config,
        "task": task.get_config(),
    }
    if dump_history:
        stats.update(
            {
                "history": histories,
                "ticks": ticks,
            }
        )
    with open(f"{output_dir}/{run_name}", "w", encoding="utf-8") as f:
        json.dump(stats, f)
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)
    p99_cost = np.percentile(costs, 99)
    p90_cost = np.percentile(costs, 90)
    logger.info(
        f"mean: {mean_cost}; std: {std_cost}; worst 1%: {p99_cost}; worst 10%: {p90_cost}"
    )
    return stats
