"""Error reporting module - self-contained."""

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


def print_error_summary(results: List[dict]) -> None:
    """Print a clear error summary at the end of the benchmark."""
    error_info = analyze_errors(results)
    
    print("\n" + "="*60)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*60)
    
    print(f"Total tasks: {error_info['total_tasks']}")
    print(f"Successful: {error_info['successful_tasks']}")
    print(f"Failed: {error_info['failed_tasks']}")
    print(f"Success rate: {error_info['success_rate']:.1f}%")
    
    # Show infeasible tasks separately if any
    if error_info['infeasible_tasks'] > 0:
        print(f"\n⚠️  INFEASIBLE TASKS: {error_info['infeasible_tasks']}")
        print("-"*40)
        print("These tasks have deadlines that are too tight for completion:")
        
        # Group infeasible tasks by deadline ratio
        infeasible_by_ratio = {}
        for task in error_info['infeasible_details']:
            ratio = task.get('deadline_ratio', 'unknown')
            if ratio not in infeasible_by_ratio:
                infeasible_by_ratio[ratio] = []
            infeasible_by_ratio[ratio].append(task)
        
        for ratio, tasks in sorted(infeasible_by_ratio.items()):
            print(f"\n  Deadline ratio {ratio}:")
            sample_task = tasks[0]
            if 'error_details' in sample_task:
                print(f"    {sample_task['error_details']}")
            print(f"    Affected: {len(tasks)} tasks")
    
    # Show trace insufficient tasks separately if any
    if error_info['trace_insufficient_tasks'] > 0:
        print(f"\n⚠️  TRACE INSUFFICIENT: {error_info['trace_insufficient_tasks']}")
        print("-"*40)
        print("These tasks have trace data shorter than the deadline period:")
        
        # Group trace insufficient tasks by scenario
        trace_insufficient_by_scenario = {}
        for task in error_info['trace_insufficient_details']:
            scenario = task.get('scenario_name', 'unknown')
            if scenario not in trace_insufficient_by_scenario:
                trace_insufficient_by_scenario[scenario] = []
            trace_insufficient_by_scenario[scenario].append(task)
        
        for scenario, tasks in sorted(trace_insufficient_by_scenario.items()):
            print(f"\n  Scenario {scenario}:")
            sample_task = tasks[0]
            if 'error_details' in sample_task:
                print(f"    {sample_task['error_details']}")
            print(f"    Affected: {len(tasks)} tasks")
    
    # Show other errors if any
    other_errors = error_info['failed_tasks'] - error_info['infeasible_tasks'] - error_info['trace_insufficient_tasks']
    if other_errors > 0:
        print(f"\n❌ OTHER ERRORS: {other_errors}")
        print("-"*40)
        
        # Group errors by strategy
        for strategy, count in error_info['errors_by_strategy'].items():
            print(f"  {strategy}: {count} failures")
        
        # Show sample error details
        print("\nSample failed tasks:")
        for i, task in enumerate(error_info['sample_failures'][:5]):
            if task.get('error_type') not in ['Task Infeasible', 'Trace Insufficient']:
                print(f"  [{i+1}] {task['strategy']} - {task['scenario_name']} - "
                      f"trace_{task['trace_index']} - {task.get('error_type', 'Unknown error')}")
    
    print("="*60 + "\n")


def analyze_errors(results: List[dict]) -> Dict:
    """Analyze results for errors and patterns."""
    total_tasks = len(results)
    failed_tasks = []
    infeasible_tasks = []
    trace_insufficient_tasks = []
    errors_by_strategy = {}
    errors_by_scenario = {}
    
    for result in results:
        if np.isnan(result.get('cost', np.nan)):
            failed_tasks.append(result)
            
            # Check if it's an infeasible task
            if result.get('error_type') == 'Task Infeasible':
                infeasible_tasks.append(result)
            elif result.get('error_type') == 'Trace Insufficient':
                trace_insufficient_tasks.append(result)
            else:
                # Only count non-infeasible and non-trace-insufficient errors in strategy breakdown
                strategy = result.get('strategy', 'unknown')
                scenario = result.get('scenario_name', 'unknown')
                
                if strategy not in errors_by_strategy:
                    errors_by_strategy[strategy] = 0
                errors_by_strategy[strategy] += 1
                
                if scenario not in errors_by_scenario:
                    errors_by_scenario[scenario] = 0
                errors_by_scenario[scenario] += 1
    
    # Get failures that are neither infeasible nor trace insufficient for sample
    other_failures = [f for f in failed_tasks if f.get('error_type') not in ['Task Infeasible', 'Trace Insufficient']]
    
    return {
        'total_tasks': total_tasks,
        'successful_tasks': total_tasks - len(failed_tasks),
        'failed_tasks': len(failed_tasks),
        'infeasible_tasks': len(infeasible_tasks),
        'infeasible_details': infeasible_tasks,
        'trace_insufficient_tasks': len(trace_insufficient_tasks),
        'trace_insufficient_details': trace_insufficient_tasks,
        'success_rate': (total_tasks - len(failed_tasks)) / total_tasks * 100 if total_tasks > 0 else 0,
        'errors_by_strategy': errors_by_strategy,
        'errors_by_scenario': errors_by_scenario,
        'sample_failures': other_failures[:10]  # First 10 other failures
    }


def log_simulation_failure(task: dict, error: Exception) -> None:
    """Log detailed information about a simulation failure."""
    logger.error(
        f"Simulation failed - Strategy: {task.get('strategy')} | "
        f"Scenario: {task.get('scenario_name')} | "
        f"Trace: {task.get('trace_index')} | "
        f"Error: {str(error)}"
    )