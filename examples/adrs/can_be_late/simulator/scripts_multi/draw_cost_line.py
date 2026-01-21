import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import List, Dict, Any

# Helper to parse the simulator log stream
def parse_log_file(file_path: str) -> List[Dict[str, Any]]:
    """Parses the log file to extract strategy decision data."""
    parsed_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if "STRATEGY_DECISION" in line:
                try:
                    json_str = line.split("STRATEGY_DECISION: ")[1]
                    data = json.loads(json_str)
                    parsed_data.append(data)
                except (IndexError, json.JSONDecodeError):
                    pass
    return parsed_data

def plot_full_analysis_with_progress_line(
    multi_data: List[Dict[str, Any]], 
    single_data: List[Dict[str, Any]], 
    total_work_needed: float, 
    deadline_seconds: float, 
    gap_seconds: float):
    """
    Plots a two-panel chart, with a unified progress line and conditional visibility.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 1, figsize=(18, 15), sharex=True)

    # --- Data preparation ---
    multi_ticks = np.array([d['tick'] for d in multi_data])
    multi_costs = np.array([d['accumulated_cost'] for d in multi_data])
    multi_work_cumulative = np.cumsum([d['last_tick_work'] for d in multi_data])

    single_ticks = np.array([d['tick'] for d in single_data])
    single_costs = np.array([d['accumulated_cost'] for d in single_data])
    single_work_cumulative = np.cumsum([d['last_tick_work'] for d in single_data])

    # ==================== Panel 1: cost vs. time ====================
    ax[0].plot(single_ticks, single_costs, label='Single-Region Strategy', color='blue', linestyle='--', linewidth=2)
    ax[0].plot(multi_ticks, multi_costs, label='Multi-Region Strategy', color='red', linewidth=2.5)
    ax[0].set_title('Part 1: Accumulated Cost vs. Time', fontsize=16)
    ax[0].set_ylabel('Accumulated Cost ($)', fontsize=14)
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    formatter = mticker.FormatStrFormatter('$%1.2f')
    ax[0].yaxis.set_major_formatter(formatter)
    ax[0].legend(fontsize=12)

    # ==================== Panel 2: progress vs. time (only when behind) ====================
    # 1. Compute the gray “on pace” guideline
    max_tick = max(multi_ticks.max(), single_ticks.max())
    unified_ticks = np.arange(0, max_tick + 1)
    # Uniform progress rate = total work / total deadline time
    progress_rate = total_work_needed / deadline_seconds
    unified_progress = progress_rate * unified_ticks * gap_seconds
    ax[1].plot(unified_ticks, unified_progress, color='gray', linestyle=':', linewidth=2.5, label='On-Pace Progress (Required)')

    # 2. Conditional rendering: only show when progress falls behind the guideline
    # Single-region strategy (blue line)
    unified_progress_for_single = progress_rate * single_ticks * gap_seconds
    blue_masked_work = np.ma.masked_where(single_work_cumulative >= unified_progress_for_single, single_work_cumulative)
    ax[1].plot(single_ticks, blue_masked_work, label='Single-Region Progress (when behind)', color='blue', linestyle='--', linewidth=2)

    # Multi-region strategy (red line)
    unified_progress_for_multi = progress_rate * multi_ticks * gap_seconds
    red_masked_work = np.ma.masked_where(multi_work_cumulative >= unified_progress_for_multi, multi_work_cumulative)
    ax[1].plot(multi_ticks, red_masked_work, label='Multi-Region Progress (when behind)', color='red', linewidth=2.5)

    # 3. Draw the total work completion line
    ax[1].axhline(y=total_work_needed, color='green', linestyle=':', linewidth=2, label=f'Total Work Required ({total_work_needed/3600:.1f}h)')

    # Axis formatting
    ax[1].set_title('Part 2: Effective Work Done vs. Time (Only showing when behind schedule)', fontsize=16)
    ax[1].set_xlabel('Time (Ticks)', fontsize=14)
    ax[1].set_ylabel('Cumulative Work Done (seconds)', fontsize=14)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].legend(fontsize=12)

    # --- Shared styling ---
    fig.suptitle('Final Analysis: Cost vs. Efficiency Debt', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("final_analysis_chart.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Replace with your own log file paths
    MULTI_REGION_LOG_PATH = 'multi_region_log.log'
    SINGLE_REGION_A_LOG_PATH = 'single_region_A_log.log'
    
    # Task parameters (match your experiment configuration)
    TOTAL_TASK_DURATION_SECONDS = 48 * 3600
    DEADLINE_SECONDS = 52 * 3600
    GAP_SECONDS = 600 # Each tick is 600 seconds
    
    multi_data = parse_log_file(MULTI_REGION_LOG_PATH)
    single_data = parse_log_file(SINGLE_REGION_A_LOG_PATH)
    
    if not multi_data or not single_data:
        print("Unable to parse enough data from the logs; check the paths.")
    else:
        plot_full_analysis_with_progress_line(
            multi_data, 
            single_data,
            TOTAL_TASK_DURATION_SECONDS,
            DEADLINE_SECONDS,
            GAP_SECONDS
        )
