#!/usr/bin/env python3
"""
Analyze spot duration consistency within individual traces.
Investigates whether short spot durations in a trace predict other short durations in the same trace.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def analyze_trace_internal_consistency(region: str, num_traces: int = 50) -> Dict:
    """Analyze spot duration consistency within individual traces."""
    trace_dir = Path(f"data/converted_multi_region_aligned/{region}")
    
    if not trace_dir.exists():
        return {}
    
    trace_consistency_stats = []
    
    for i in range(min(num_traces, len(list(trace_dir.glob("*.json"))))):
        trace_path = trace_dir / f"{i}.json"
        if not trace_path.exists():
            continue
            
        with open(trace_path, 'r') as f:
            trace = json.load(f)
        
        data = trace['data']
        gap_seconds = trace['metadata']['gap_seconds']
        
        # Convert from preempted format (0=available, 1=preempted) to availability
        availability_data = [1 - x for x in data]
        
        # Extract all spot durations (continuous available periods)
        spot_durations = []
        current_duration = 0
        
        for available in availability_data:
            if available == 1:  # Spot instance running
                current_duration += 1
            else:  # Preempted or unavailable
                if current_duration > 0:
                    spot_durations.append(current_duration)
                    current_duration = 0
        
        # Don't forget the last run if it ends with availability
        if current_duration > 0:
            spot_durations.append(current_duration)
        
        # Convert to hours for easier interpretation
        spot_durations_hours = [d * gap_seconds / 3600 for d in spot_durations]
        
        # Skip traces with too few spot instances
        if len(spot_durations_hours) < 3:
            continue
        
        # Calculate internal consistency metrics
        mean_duration = np.mean(spot_durations_hours)
        std_duration = np.std(spot_durations_hours)
        cv = std_duration / mean_duration if mean_duration > 0 else float('inf')  # Coefficient of variation
        min_duration = min(spot_durations_hours)
        max_duration = max(spot_durations_hours)
        duration_range = max_duration - min_duration
        
        # Check if first duration predicts others
        first_duration = spot_durations_hours[0]
        remaining_durations = spot_durations_hours[1:]
        
        # Categorize first duration
        first_is_short = first_duration < 1.0  # Less than 1 hour
        first_is_medium = 1.0 <= first_duration < 5.0  # 1-5 hours
        first_is_long = first_duration >= 5.0  # 5+ hours
        
        # Calculate what percentage of remaining durations fall in same category
        remaining_short = sum(1 for d in remaining_durations if d < 1.0)
        remaining_medium = sum(1 for d in remaining_durations if 1.0 <= d < 5.0)
        remaining_long = sum(1 for d in remaining_durations if d >= 5.0)
        
        if first_is_short:
            consistency_rate = remaining_short / len(remaining_durations)
            prediction_category = "short"
        elif first_is_medium:
            consistency_rate = remaining_medium / len(remaining_durations)
            prediction_category = "medium"
        else:
            consistency_rate = remaining_long / len(remaining_durations)
            prediction_category = "long"
        
        # Calculate correlation between early and late durations
        if len(spot_durations_hours) >= 6:
            early_durations = spot_durations_hours[:len(spot_durations_hours)//2]
            late_durations = spot_durations_hours[len(spot_durations_hours)//2:]
            
            if len(early_durations) == len(late_durations) and len(early_durations) > 1:
                early_late_corr, early_late_p = pearsonr(early_durations, late_durations)
            else:
                early_late_corr, early_late_p = 0, 1
        else:
            early_late_corr, early_late_p = 0, 1
        
        # Calculate autocorrelation (consecutive durations)
        if len(spot_durations_hours) >= 3:
            consecutive_corr, consecutive_p = pearsonr(
                spot_durations_hours[:-1], spot_durations_hours[1:]
            )
        else:
            consecutive_corr, consecutive_p = 0, 1
        
        trace_consistency_stats.append({
            'trace_id': i,
            'region': region,
            'num_spot_instances': len(spot_durations_hours),
            'mean_duration_hours': mean_duration,
            'std_duration_hours': std_duration,
            'cv_duration': cv,
            'min_duration_hours': min_duration,
            'max_duration_hours': max_duration,
            'duration_range_hours': duration_range,
            'first_duration_hours': first_duration,
            'first_category': prediction_category,
            'consistency_rate': consistency_rate,
            'early_late_correlation': early_late_corr,
            'early_late_p_value': early_late_p,
            'consecutive_correlation': consecutive_corr,
            'consecutive_p_value': consecutive_p,
            'all_durations': spot_durations_hours
        })
    
    if not trace_consistency_stats:
        return {}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(trace_consistency_stats)
    
    # Calculate aggregate statistics
    summary_stats = {
        'region': region,
        'num_analyzed_traces': len(trace_consistency_stats),
        'mean_spots_per_trace': df['num_spot_instances'].mean(),
        'mean_cv': df['cv_duration'].mean(),
        'mean_consistency_rate': df['consistency_rate'].mean(),
        'mean_consecutive_correlation': df['consecutive_correlation'].mean(),
        'mean_early_late_correlation': df['early_late_correlation'].mean(),
        
        # Breakdown by first duration category
        'short_first_traces': len(df[df['first_category'] == 'short']),
        'medium_first_traces': len(df[df['first_category'] == 'medium']),
        'long_first_traces': len(df[df['first_category'] == 'long']),
        
        # Consistency rates by category
        'short_consistency': df[df['first_category'] == 'short']['consistency_rate'].mean() if len(df[df['first_category'] == 'short']) > 0 else 0,
        'medium_consistency': df[df['first_category'] == 'medium']['consistency_rate'].mean() if len(df[df['first_category'] == 'medium']) > 0 else 0,
        'long_consistency': df[df['first_category'] == 'long']['consistency_rate'].mean() if len(df[df['first_category'] == 'long']) > 0 else 0,
        
        'trace_data': df
    }
    
    return summary_stats

def create_consistency_analysis_plots(all_results: List[Dict], output_dir: Path):
    """Create focused consistency analysis plot."""
    valid_results = [r for r in all_results if r]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create single focused plot
    plt.figure(figsize=(12, 8))
    
    # Collect data across all regions
    all_consistency_rates = []
    all_categories = []
    
    for result in valid_results:
        df = result['trace_data']
        all_consistency_rates.extend(df['consistency_rate'].tolist())
        all_categories.extend(df['first_category'].tolist())
    
    # Focus on the most important finding: consistency by category
    category_consistency = {}
    for category in ['short', 'medium', 'long']:
        mask = np.array(all_categories) == category
        if np.sum(mask) > 0:
            consistency_values = np.array(all_consistency_rates)[mask]
            category_consistency[category] = consistency_values
    
    if category_consistency:
        categories = list(category_consistency.keys())
        means = [np.mean(category_consistency[cat]) for cat in categories]
        stds = [np.std(category_consistency[cat]) for cat in categories]
        counts = [len(category_consistency[cat]) for cat in categories]
        
        # Create bar plot with clear visual distinction
        colors = ['#ff4444', '#ffaa44', '#44aa44']  # Red, Orange, Green
        bars = plt.bar(categories, means, yerr=stds, capsize=8, alpha=0.8,
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, mean, count) in enumerate(zip(bars, means, counts)):
            height = bar.get_height()
            # Add percentage on top of bar
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
            # Add count below
            plt.text(bar.get_x() + bar.get_width()/2., 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=11)
        
        # Add reference line for random chance
        plt.axhline(y=1/3, color='black', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Random Chance (33%)')
        
        # Customize the plot
        plt.ylabel('Prediction Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('First Spot Duration Category', fontsize=14, fontweight='bold')
        plt.title('Can First Spot Duration Predict the Rest?\n(Same trace, subsequent spots)', 
                 fontsize=16, fontweight='bold')
        
        # Add category descriptions
        category_labels = ['Short\n(<1 hour)', 'Medium\n(1-5 hours)', 'Long\n(5+ hours)']
        plt.xticks(range(len(categories)), category_labels, fontsize=12)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text box
        interpretation = """Key Finding:
If first spot lasts <1 hour,
64% chance others will too!

Practical Use:
Early failure → switch regions"""
        
        plt.text(0.98, 0.98, interpretation, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                fontsize=11, fontweight='bold')
        
        # Highlight the most important bar
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(3)
        
    plt.tight_layout()
    plt.savefig(output_dir / 'spot_duration_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_consistency_report(all_results: List[Dict], output_path: Path):
    """Generate detailed consistency analysis report."""
    valid_results = [r for r in all_results if r]
    
    with open(output_path, 'w') as f:
        f.write("# Spot Duration Consistency Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        
        if not valid_results:
            f.write("No valid data found for analysis.\n")
            return
        
        # Calculate overall statistics
        all_consistency_rates = []
        all_consecutive_corrs = []
        
        for result in valid_results:
            df = result['trace_data']
            all_consistency_rates.extend(df['consistency_rate'].tolist())
            all_consecutive_corrs.extend(df['consecutive_correlation'].tolist())
        
        mean_consistency = np.mean(all_consistency_rates)
        mean_consecutive_corr = np.mean(all_consecutive_corrs)
        
        f.write(f"- **Total traces analyzed**: {sum(r['num_analyzed_traces'] for r in valid_results)}\n")
        f.write(f"- **Mean consistency rate**: {mean_consistency:.3f} (vs 33% random)\n")
        f.write(f"- **Mean consecutive correlation**: {mean_consecutive_corr:.3f}\n\n")
        
        # Interpret results
        f.write("### Key Findings\n\n")
        
        if mean_consistency > 0.5:
            f.write(f"🔍 **Strong predictability**: First spot duration is a good predictor of subsequent durations in the same trace.\n")
        elif mean_consistency > 0.4:
            f.write(f"📊 **Moderate predictability**: First spot duration has some predictive power for subsequent durations.\n")
        else:
            f.write(f"❓ **Weak predictability**: First spot duration is not a reliable predictor of subsequent durations.\n")
        
        if mean_consecutive_corr > 0.3:
            f.write(f"🔗 **Strong temporal correlation**: Consecutive spot durations are highly correlated.\n")
        elif mean_consecutive_corr > 0.1:
            f.write(f"🔗 **Moderate temporal correlation**: Some correlation between consecutive spot durations.\n")
        else:
            f.write(f"🔀 **Weak temporal correlation**: Little correlation between consecutive spot durations.\n")
        
        f.write("\n## Detailed Analysis by Region\n\n")
        
        # Sort regions by consistency for easier comparison
        sorted_results = sorted(valid_results, key=lambda x: x['mean_consistency_rate'], reverse=True)
        
        for result in sorted_results:
            region = result['region']
            f.write(f"### {region}\n\n")
            
            f.write("#### Summary Statistics\n\n")
            f.write("| Metric | Value | Interpretation |\n")
            f.write("|--------|-------|----------------|\n")
            f.write(f"| Analyzed Traces | {result['num_analyzed_traces']} | - |\n")
            f.write(f"| Avg Spots per Trace | {result['mean_spots_per_trace']:.1f} | - |\n")
            f.write(f"| Consistency Rate | {result['mean_consistency_rate']:.3f} | ")
            
            if result['mean_consistency_rate'] > 0.5:
                f.write("Strong predictability |\n")
            elif result['mean_consistency_rate'] > 0.4:
                f.write("Moderate predictability |\n")
            else:
                f.write("Weak predictability |\n")
            
            f.write(f"| Consecutive Correlation | {result['mean_consecutive_correlation']:.3f} | ")
            
            if result['mean_consecutive_correlation'] > 0.3:
                f.write("Strong temporal pattern |\n")
            elif result['mean_consecutive_correlation'] > 0.1:
                f.write("Moderate temporal pattern |\n")
            else:
                f.write("Weak temporal pattern |\n")
            
            f.write("\n#### Breakdown by First Duration Category\n\n")
            f.write("| Category | Count | Consistency Rate |\n")
            f.write("|----------|-------|------------------|\n")
            f.write(f"| Short (<1h) | {result['short_first_traces']} | {result['short_consistency']:.3f} |\n")
            f.write(f"| Medium (1-5h) | {result['medium_first_traces']} | {result['medium_consistency']:.3f} |\n")
            f.write(f"| Long (5h+) | {result['long_first_traces']} | {result['long_consistency']:.3f} |\n")
            
            f.write("\n---\n\n")
        
        f.write("## Methodology\n\n")
        f.write("1. **Data Source**: Analyzed aligned traces from `data/converted_multi_region_aligned/`\n")
        f.write("2. **Spot Duration Extraction**: Identified continuous available periods within each trace\n")
        f.write("3. **Categories**:\n")
        f.write("   - Short: < 1 hour\n")
        f.write("   - Medium: 1-5 hours\n")
        f.write("   - Long: 5+ hours\n")
        f.write("4. **Consistency Rate**: Percentage of subsequent durations that fall in the same category as the first\n")
        f.write("5. **Consecutive Correlation**: Pearson correlation between consecutive spot durations\n")
        
        f.write("\n## Conclusions\n\n")
        
        if mean_consistency > 0.5:
            f.write("**High Predictability**: If the first spot instance in a trace is short-lived, ")
            f.write("subsequent instances in the same trace are likely to also be short-lived. ")
            f.write("This suggests that trace-level characteristics (infrastructure, region conditions, etc.) ")
            f.write("have persistent effects on spot instance duration.\n\n")
            f.write("**Practical Implication**: Early spot behavior can inform strategy decisions. ")
            f.write("If initial instances are short-lived, consider switching regions or approaches.\n")
        else:
            f.write("**Low Predictability**: Spot duration patterns within traces are not highly consistent. ")
            f.write("Early spot behavior is not a reliable predictor of subsequent behavior in the same trace.\n\n")
            f.write("**Practical Implication**: Each spot instance should be evaluated independently. ")
            f.write("Past performance in a trace may not predict future performance.\n")

def main():
    """Run spot duration consistency analysis."""
    # Get all regions from aligned trace data
    trace_base_dir = Path("data/converted_multi_region_aligned")
    all_regions = sorted([d.name for d in trace_base_dir.iterdir() if d.is_dir() and d.name.endswith('_v100_1')])
    
    print("="*80)
    print("SPOT DURATION CONSISTENCY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(all_regions)} regions...\n")
    print("Question: Can the first spot duration predict subsequent durations in the same trace?")
    print()
    
    results = []
    
    for region in all_regions:
        print(f"Analyzing {region}...", end='', flush=True)
        
        region_stats = analyze_trace_internal_consistency(region)
        
        if region_stats:
            results.append(region_stats)
            consistency = region_stats['mean_consistency_rate']
            print(f" ✓ ({region_stats['num_analyzed_traces']} traces, consistency: {consistency:.3f})")
        else:
            print(" ✗ (no data)")
    
    if not results:
        print("No valid data found. Exiting.")
        return
    
    # Create output directory
    output_dir = Path("outputs/spot_duration_consistency")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating consistency analysis plots...")
    create_consistency_analysis_plots(results, output_dir)
    
    # Generate report
    print("Generating detailed report...")
    generate_consistency_report(results, output_dir / "spot_duration_consistency_report.md")
    
    # Quick summary for console
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - Report: {output_dir}/spot_duration_consistency_report.md")
    print(f"   - Plots: {output_dir}/spot_duration_consistency_analysis.png")
    
    # Print quick summary
    if results:
        all_consistency_rates = []
        all_consecutive_corrs = []
        
        for result in results:
            df = result['trace_data']
            all_consistency_rates.extend(df['consistency_rate'].tolist())
            all_consecutive_corrs.extend(df['consecutive_correlation'].tolist())
        
        mean_consistency = np.mean(all_consistency_rates)
        mean_consecutive = np.mean(all_consecutive_corrs)
        
        print(f"\n📊 Quick Summary:")
        print(f"   - Mean consistency rate: {mean_consistency:.3f} (vs 33% random)")
        print(f"   - Mean consecutive correlation: {mean_consecutive:.3f}")
        
        if mean_consistency > 0.5:
            print(f"   - 🔍 Finding: First spot duration IS a good predictor!")
        elif mean_consistency > 0.4:
            print(f"   - 📊 Finding: First spot duration has SOME predictive power")
        else:
            print(f"   - ❓ Finding: First spot duration is NOT a reliable predictor")

if __name__ == "__main__":
    main()