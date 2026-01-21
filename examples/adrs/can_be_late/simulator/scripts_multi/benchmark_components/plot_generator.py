"""Plot generation module - self-contained."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)

# Visual configuration
STRATEGY_DISPLAY_NAMES = {
    "multi_region_rc_cr_threshold": "Multi-Region Uniform Progress",
    "multi_region_rc_cr_no_cond2": "Multi-Region Uniform Progress (w/o Cond. 2)",
    "multi_region_rc_cr_randomized": "Multi-Region Uniform Progress (Random)",
    "multi_region_rc_cr_reactive": "Multi-Region Uniform Progress (Reactive)",
    "lazy_cost_aware_multi": "Lazy Cost-Aware Multi",
    "evolutionary_simple_v2": "Evolutionary Simple v2",
    "rc_cr_threshold": "Single-Region Uniform Progress",
    "quick_optimal": "Optimal (on Union)",
    # Trace mode baselines
    "rc_cr_threshold_best_single": "RC/CR Threshold (Best Single)",
    "rc_cr_threshold_average_single": "RC/CR Threshold (Average Single)",
    "quick_optimal_best_single": "Optimal (Best Single)",
    "quick_optimal_average_single": "Optimal (Average Single)",
    "best_110": "Best 110",
    "best_210": "Best 210",
}

STRATEGY_COLORS = {
    "multi_region_rc_cr_threshold": "tab:blue",
    "multi_region_rc_cr_no_cond2": "tab:purple",
    "multi_region_rc_cr_randomized": "tab:cyan",
    "multi_region_rc_cr_reactive": "tab:brown",
    "lazy_cost_aware_multi": "tab:green",
    "evolutionary_simple_v2": "tab:red",
    "rc_cr_threshold": "tab:gray",
    "quick_optimal": "tab:orange",
    # Trace mode baselines
    "rc_cr_threshold_best_single": "darkgray",
    "rc_cr_threshold_average_single": "lightgray",
    "quick_optimal_best_single": "darkorange",
    "quick_optimal_average_single": "lightsalmon",
    "best_110": "tab:pink",
    "best_210": "tab:olive",
}


def create_restart_overhead_plot(
    df: pd.DataFrame, 
    num_traces: int, 
    restart_overheads: List[float],
    output_dir: Path
) -> Path:
    """Generate restart overhead impact analysis plot."""
    logger.info(f"📊 Creating restart overhead analysis plot with {len(restart_overheads)} values")
    
    # Select up to 4 scenarios with the most data points
    scenario_counts = df['scenario_name'].value_counts()
    available_scenarios = list(scenario_counts.head(4).index)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(available_scenarios[:4]):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Plot each strategy
        for strategy in sorted(scenario_df['strategy'].unique()):
            strategy_data = scenario_df[scenario_df['strategy'] == strategy]
            
            # Group by restart_overhead
            overhead_costs = []
            for ro in restart_overheads:
                ro_data = strategy_data[strategy_data['restart_overhead'] == ro]
                if not ro_data.empty:
                    overhead_costs.append(ro_data['cost'].mean())
                else:
                    overhead_costs.append(np.nan)
            
            # Plot if we have data
            if any(not np.isnan(c) for c in overhead_costs):
                marker = 's' if strategy == 'rc_cr_threshold' else 'o'
                linestyle = '--' if strategy == 'rc_cr_threshold' else '-'
                ax.plot(restart_overheads, overhead_costs,
                       marker=marker, linewidth=2, markersize=6, linestyle=linestyle,
                       color=STRATEGY_COLORS.get(strategy, 'gray'),
                       label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy))
        
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xlabel('Restart Overhead (hours)', fontsize=11)
        ax.set_ylabel('Mean Execution Cost ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(restart_overheads)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(len(available_scenarios), 4):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    if available_scenarios:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.suptitle(f'Restart Overhead Impact Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
    plot_path = output_dir / f"restart_overhead_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Restart overhead analysis plot saved to {plot_path}")
    return plot_path


def create_checkpoint_size_plot(
    df: pd.DataFrame,
    num_traces: int,
    checkpoint_sizes: List[float],
    output_dir: Path
) -> Path:
    """Generate checkpoint size impact analysis plot."""
    logger.info(f"📊 Creating checkpoint size analysis plot with {len(checkpoint_sizes)} sizes")
    
    # Select up to 4 scenarios with the most data points
    scenario_counts = df['scenario_name'].value_counts()
    available_scenarios = list(scenario_counts.head(4).index)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(available_scenarios[:4]):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Plot each strategy
        for strategy in sorted(scenario_df['strategy'].unique()):
            strategy_data = scenario_df[scenario_df['strategy'] == strategy]
            
            # Group by checkpoint_size
            size_costs = []
            for cs in checkpoint_sizes:
                cs_data = strategy_data[strategy_data['checkpoint_size'] == cs]
                if not cs_data.empty:
                    size_costs.append(cs_data['cost'].mean())
                else:
                    size_costs.append(np.nan)
            
            # Plot if we have data
            if any(not np.isnan(c) for c in size_costs):
                marker = 's' if strategy == 'rc_cr_threshold' else 'o'
                linestyle = '--' if strategy == 'rc_cr_threshold' else '-'
                ax.plot(checkpoint_sizes, size_costs,
                       marker=marker, linewidth=2, markersize=6, linestyle=linestyle,
                       color=STRATEGY_COLORS.get(strategy, 'gray'),
                       label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy))
        
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_xlabel('Checkpoint Size (GB)', fontsize=11)
        ax.set_ylabel('Mean Execution Cost ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(checkpoint_sizes)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(len(available_scenarios), 4):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    if available_scenarios:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.suptitle(f'Checkpoint Size Impact Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
    plot_path = output_dir / f"checkpoint_size_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Checkpoint size analysis plot saved to {plot_path}")
    return plot_path


def create_scenario_bar_plot(
    df: pd.DataFrame,
    num_traces: int,
    output_dir: Path,
    scenario_configs: List[Dict] = None
) -> Path:
    """Generate main comparison bar plot across scenarios with trace statistics."""
    logger.info("📊 Creating scenario comparison bar plot with trace statistics")
    
    # Parse cost column (convert from string list to numeric)
    def parse_cost(cost_str):
        try:
            if isinstance(cost_str, str) and cost_str.startswith('['):
                # Parse string representation of list and take first element
                import ast
                cost_list = ast.literal_eval(cost_str)
                return float(cost_list[0]) if isinstance(cost_list, list) and len(cost_list) > 0 else float('nan')
            else:
                return float(cost_str)
        except:
            return float('nan')
    
    df['cost_numeric'] = df['cost'].apply(parse_cost)
    
    # Group by scenario and strategy, calculate mean costs
    grouped = df.groupby(['scenario_name', 'strategy'])['cost_numeric'].mean().reset_index()
    grouped.rename(columns={'cost_numeric': 'cost'}, inplace=True)
    
    # Get unique scenarios and strategies
    scenarios = grouped['scenario_name'].unique()
    strategies = sorted(grouped['strategy'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(scenarios))
    width = 0.8 / len(strategies)
    
    # Plot bars for each strategy
    for i, strategy in enumerate(strategies):
        strategy_data = grouped[grouped['strategy'] == strategy]
        costs = []
        for scenario in scenarios:
            scenario_cost = strategy_data[strategy_data['scenario_name'] == scenario]['cost']
            costs.append(scenario_cost.values[0] if len(scenario_cost) > 0 else np.nan)
        
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, costs, width, 
                      label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                      color=STRATEGY_COLORS.get(strategy, 'gray'),
                      alpha=0.8)
        
        # Add value labels on bars
        for bar, cost in zip(bars, costs):
            if not np.isnan(cost):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${cost:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Scenarios', fontsize=12)
    ax.set_ylabel('Mean Execution Cost ($)', fontsize=12)
    ax.set_title(f'Strategy Performance Across Different Region Configurations (n={num_traces})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add trace availability statistics for each scenario
    if scenario_configs:
        # Find y position for text (below the x-axis)
        y_min = ax.get_ylim()[0]
        y_text = y_min - (ax.get_ylim()[1] - y_min) * 0.15
        
        for i, scenario_name in enumerate(scenarios):
            # Find matching scenario config
            config = next((s for s in scenario_configs if s['name'] == scenario_name), None)
            if config:
                stats = calculate_scenario_trace_statistics(scenario_name, config['regions'])
                if stats['overall']:
                    text = f"Avg Avail: {stats['overall']['mean']:.1%}\n({len(config['regions'])} regions)"
                    ax.text(i, y_text, text, ha='center', va='top', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    plot_path = output_dir / f"scenario_comparison_bar_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Scenario comparison bar plot saved to {plot_path}")
    return plot_path


def create_deadline_sensitivity_plot(
    df: pd.DataFrame,
    num_traces: int,
    deadline_ratios: List[float],
    output_dir: Path
) -> Path:
    """Generate deadline sensitivity analysis with dynamic subplots."""
    logger.info("📊 Creating deadline sensitivity analysis plot")
    
    # Get unique scenarios
    scenarios = sorted(df['scenario_name'].unique())
    n_scenarios = len(scenarios)
    
    # Determine subplot layout
    if n_scenarios <= 4:
        rows, cols = 2, 2
    elif n_scenarios <= 6:
        rows, cols = 2, 3
    elif n_scenarios <= 9:
        rows, cols = 3, 3
    else:
        rows = int(np.ceil(np.sqrt(n_scenarios)))
        cols = int(np.ceil(n_scenarios / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        scenario_df = df[df['scenario_name'] == scenario]
        
        # Define markers and linestyles for different strategies
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        # Plot each strategy
        for i, strategy in enumerate(sorted(scenario_df['strategy'].unique())):
            strategy_data = scenario_df[scenario_df['strategy'] == strategy]
            
            # Group by deadline_ratio
            grouped = strategy_data.groupby('deadline_ratio')['cost'].agg(['mean', 'std', 'count'])
            
            x_values = grouped.index
            y_values = grouped['mean']
            
            # Add small y-offset to avoid complete overlap
            # Group strategies by similar cost ranges
            if strategy in ['multi_region_rc_cr_threshold', 'multi_region_rc_cr_no_cond2', 
                          'multi_region_rc_cr_randomized', 'multi_region_rc_cr_reactive']:
                y_offset = i * 0.3  # Small offset for overlapping multi-region strategies
            else:
                y_offset = 0
            
            ax.plot(x_values, y_values + y_offset,
                   marker=markers[i % len(markers)], 
                   linewidth=2.5, 
                   markersize=8, 
                   linestyle=linestyles[i % len(linestyles)],
                   color=STRATEGY_COLORS.get(strategy, 'gray'),
                   label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                   alpha=0.85,  # Slight transparency
                   markeredgecolor='white',  # White edge on markers for visibility
                   markeredgewidth=0.5)
        
        ax.set_title(scenario, fontsize=11, fontweight='bold')
        ax.set_xlabel('Deadline Ratio', fontsize=10)
        ax.set_ylabel('Mean Cost ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add vertical line at ratio=1.0
        ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
        
        # Adjust Y-axis
        _adjust_y_axis(ax, scenario_df)
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].set_visible(False)
    
    # Add a single legend for all subplots
    # Get handles and labels from the first subplot that has data
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    plt.suptitle(f'Deadline Sensitivity Analysis (n={num_traces})', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for legend
    
    plot_path = output_dir / f"deadline_sensitivity_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Deadline sensitivity plot saved to {plot_path}")
    return plot_path


def create_cost_heatmap(
    df: pd.DataFrame,
    num_traces: int,
    deadline_ratios: List[float],
    checkpoint_sizes: List[float],
    output_dir: Path
) -> Path:
    """Generate cost heatmap for deadline × checkpoint interactions."""
    logger.info("📊 Creating cost heatmap (deadline × checkpoint)")
    
    # Get unique scenarios and strategies
    scenarios = sorted(df['scenario_name'].unique())
    strategies = sorted(df['strategy'].unique())
    
    # Calculate subplot layout
    total_subplots = len(scenarios) * len(strategies)
    cols = min(4, len(strategies))  # Max 4 columns
    rows = int(np.ceil(total_subplots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    subplot_idx = 0
    for scenario in scenarios:
        for strategy in strategies:
            if subplot_idx >= len(axes):
                break
                
            ax = axes[subplot_idx]
            
            # Filter data
            subset = df[(df['scenario_name'] == scenario) & (df['strategy'] == strategy)]
            
            # Create pivot table for heatmap
            pivot = subset.pivot_table(
                values='cost',
                index='checkpoint_size',
                columns='deadline_ratio',
                aggfunc='mean'
            )
            
            # Create heatmap
            if not pivot.empty:
                im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
                
                # Set ticks
                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns], fontsize=8)
                ax.set_yticklabels([f'{int(y)}GB' for y in pivot.index], fontsize=8)
                
                # Add text annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        value = pivot.values[i, j]
                        if not np.isnan(value):
                            text = ax.text(j, i, f'${value:.0f}',
                                         ha="center", va="center", color="black", fontsize=7)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                
                ax.set_title(f'{scenario}\n{STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}', 
                           fontsize=9, fontweight='bold')
                ax.set_xlabel('Deadline Ratio', fontsize=8)
                ax.set_ylabel('Checkpoint Size', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scenario}\n{STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}', 
                           fontsize=9, fontweight='bold')
            
            subplot_idx += 1
    
    # Hide unused subplots
    for i in range(subplot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Cost Heatmap: Deadline × Checkpoint Interaction (n={num_traces})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / f"cost_heatmap_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Cost heatmap saved to {plot_path}")
    return plot_path


def create_region_scaling_plot(
    df: pd.DataFrame,
    num_traces: int,
    output_dir: Path
) -> Path:
    """Generate region scaling benefit analysis."""
    logger.info("📊 Creating region scaling analysis plot")
    
    # Use num_regions field directly
    df = df.copy()
    df = df[df['num_regions'] > 0]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot for each strategy
    plotted_strategies = []
    for strategy in sorted(df['strategy'].unique()):
        strategy_data = df[df['strategy'] == strategy]
        
        # Skip if no valid data for this strategy
        if strategy_data.empty or strategy_data['cost'].isna().all():
            logger.warning(f"No valid data for strategy {strategy}, skipping")
            continue
        
        # Group by number of regions
        grouped = strategy_data.groupby('num_regions')['cost'].agg(['mean', 'std', 'count'])
        
        # Skip if no groups formed
        if grouped.empty:
            logger.warning(f"No region groups formed for strategy {strategy}, skipping")
            continue
        
        x_values = grouped.index
        y_values = grouped['mean']
        yerr = grouped['std'] / np.sqrt(grouped['count'])  # Standard error
        
        marker = 's' if strategy == 'rc_cr_threshold' else 'o'
        linestyle = '--' if strategy == 'rc_cr_threshold' else '-'
        
        ax.errorbar(x_values, y_values, yerr=yerr,
                   marker=marker, linewidth=2, markersize=8, linestyle=linestyle,
                   color=STRATEGY_COLORS.get(strategy, 'gray'),
                   label=STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                   capsize=5, alpha=0.8)
        plotted_strategies.append(strategy)
    
    ax.set_xlabel('Number of Regions', fontsize=12)
    ax.set_ylabel('Mean Execution Cost ($)', fontsize=12)
    ax.set_title(f'Region Scaling Benefits Analysis (n={num_traces})', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([2, 3, 5])
    ax.grid(True, alpha=0.3)
    
    # Only add legend if we actually plotted something
    if plotted_strategies:
        ax.legend(fontsize=10, loc='best')
    else:
        # Add text indicating no data available
        ax.text(0.5, 0.5, 'No valid data available for region scaling analysis', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Add percentage improvement annotations
    # Calculate improvement from 2 to 5 regions for each strategy
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        costs_by_region = strategy_data.groupby('num_regions')['cost'].mean()
        
        if 2 in costs_by_region.index and 5 in costs_by_region.index:
            cost_2r = costs_by_region[2]
            cost_5r = costs_by_region[5]
            improvement = (cost_2r - cost_5r) / cost_2r * 100
            
            # Add text annotation
            ax.text(0.02, 0.98 - 0.05 * list(df['strategy'].unique()).index(strategy), 
                   f'{STRATEGY_DISPLAY_NAMES.get(strategy, strategy)}: {improvement:.1f}% improvement (2→5 regions)',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    plot_path = output_dir / f"region_scaling_analysis_t{num_traces}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Region scaling analysis plot saved to {plot_path}")
    return plot_path


def calculate_scenario_trace_statistics(scenario_name: str, regions: List[str], 
                                       data_path: str = "data/converted_multi_region_aligned") -> Dict:
    """Calculate trace availability statistics for a scenario."""
    stats = {
        'scenario': scenario_name,
        'regions': {},
        'overall': {}
    }
    
    all_availabilities = []
    
    # Check if base data path exists
    base_path = Path(data_path)
    if not base_path.exists():
        logger.warning(f"Data path does not exist: {base_path.absolute()}")
        return stats
    
    for region in regions:
        region_path = base_path / region
        if not region_path.exists():
            logger.warning(f"Region path does not exist: {region_path.absolute()}")
            continue
            
        region_availabilities = []
        
        # Analyze first 10 traces (or all if less)
        trace_files = sorted(region_path.glob("*.json"))[:10]
        
        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                # Convert from preempted format (0=available, 1=preempted)
                availability = 1 - np.mean(trace_data['data'])
                region_availabilities.append(availability)
                all_availabilities.append(availability)
            except:
                continue
        
        if region_availabilities:
            stats['regions'][region] = {
                'mean': np.mean(region_availabilities),
                'std': np.std(region_availabilities),
                'min': np.min(region_availabilities),
                'max': np.max(region_availabilities)
            }
    
    if all_availabilities:
        stats['overall'] = {
            'mean': np.mean(all_availabilities),
            'std': np.std(all_availabilities),
            'min': np.min(all_availabilities),
            'max': np.max(all_availabilities),
            'num_traces': len(all_availabilities)
        }
    
    return stats


def create_scenario_availability_plot(
    scenario_configs: List[Dict],
    output_dir: Path
) -> Path:
    """Create a detailed plot showing trace availability statistics for all scenarios."""
    logger.info("📊 Creating scenario availability statistics plot")
    
    # Calculate statistics for each scenario
    all_stats = []
    for config in scenario_configs:
        stats = calculate_scenario_trace_statistics(config['name'], config['regions'])
        if stats['overall']:
            all_stats.append(stats)
    
    if not all_stats:
        logger.warning("No availability statistics available - checked data path: data/converted_multi_region_aligned")
        # Add diagnostic info
        data_path = Path("data/converted_multi_region_aligned")
        if data_path.exists():
            available_regions = [d.name for d in data_path.iterdir() if d.is_dir()]
            logger.info(f"Available regions in data path: {available_regions}")
        else:
            logger.error(f"Data path does not exist: {data_path.absolute()}")
        return None
    
    # Create figure with subplots for each scenario
    n_scenarios = len(all_stats)
    rows = int(np.ceil(np.sqrt(n_scenarios)))
    cols = int(np.ceil(n_scenarios / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_scenarios == 1:
        axes = [axes]
    elif n_scenarios > 1:
        axes = axes.flatten()
    
    for idx, stats in enumerate(all_stats):
        ax = axes[idx]
        
        # Prepare data for plotting
        regions = list(stats['regions'].keys())
        region_means = [stats['regions'][r]['mean'] for r in regions]
        region_stds = [stats['regions'][r]['std'] for r in regions]
        
        # Short region names for x-axis
        short_regions = [r.split('_')[0].replace('us-', '') for r in regions]
        
        # Create bar plot with error bars
        x = np.arange(len(regions))
        bars = ax.bar(x, region_means, yerr=region_stds, capsize=5, 
                      alpha=0.8, color='skyblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, mean in zip(bars, region_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{mean:.1%}', ha='center', va='bottom', fontsize=8)
        
        # Add overall average line
        ax.axhline(stats['overall']['mean'], color='red', linestyle='--', 
                  label=f"Overall: {stats['overall']['mean']:.1%}")
        
        ax.set_xlabel('Region', fontsize=10)
        ax.set_ylabel('Availability Rate', fontsize=10)
        ax.set_title(f"{stats['scenario']}\n(n={stats['overall']['num_traces']} traces)", 
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_regions, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_scenarios, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Trace Availability Statistics by Scenario', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "scenario_availability_statistics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Scenario availability statistics plot saved to {plot_path}")
    return plot_path


def _adjust_y_axis(ax, scenario_df: pd.DataFrame) -> None:
    """Helper to adjust Y-axis range for better visibility."""
    if not scenario_df.empty:
        all_costs = scenario_df['cost'].dropna()
        if len(all_costs) > 0:
            cost_min, cost_max = all_costs.min(), all_costs.max()
            cost_range = cost_max - cost_min
            if cost_range > 0:
                margin = cost_range * 0.1
                ax.set_ylim(cost_min - margin, cost_max + margin)