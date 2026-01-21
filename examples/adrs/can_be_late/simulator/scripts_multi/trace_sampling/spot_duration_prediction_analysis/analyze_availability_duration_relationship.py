#!/usr/bin/env python3
"""
Analyze the relationship between availability rate and spot instance duration patterns.
Investigates whether regions with poor availability also have longer/shorter spot durations.
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

def analyze_availability_and_durations(region: str, num_traces: int = 100) -> Dict:
    """Analyze availability rate and spot duration patterns for a region."""
    trace_dir = Path(f"data/converted_multi_region_aligned/{region}")
    
    if not trace_dir.exists():
        return {}
    
    all_trace_stats = []
    
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
        
        # Calculate basic availability metrics
        availability_rate = sum(availability_data) / len(availability_data) if availability_data else 0
        
        # Analyze spot instance durations (continuous available periods)
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
        
        # Analyze preemption gaps (continuous unavailable periods)
        preemption_gaps = []
        current_gap = 0
        
        for available in availability_data:
            if available == 0:  # Unavailable
                current_gap += 1
            else:  # Available
                if current_gap > 0:
                    preemption_gaps.append(current_gap)
                    current_gap = 0
        
        # Don't forget the last gap if it ends with unavailability
        if current_gap > 0:
            preemption_gaps.append(current_gap)
        
        # Calculate duration statistics
        if spot_durations:
            spot_duration_stats = {
                'mean': np.mean(spot_durations),
                'median': np.median(spot_durations),
                'std': np.std(spot_durations),
                'min': min(spot_durations),
                'max': max(spot_durations),
                'count': len(spot_durations),
                'total_time': sum(spot_durations)
            }
            # Convert to hours for easier interpretation
            spot_duration_hours = {
                k: v * gap_seconds / 3600 if k in ['mean', 'median', 'std', 'min', 'max', 'total_time'] else v 
                for k, v in spot_duration_stats.items()
            }
        else:
            spot_duration_hours = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0, 'total_time': 0
            }
        
        if preemption_gaps:
            gap_stats = {
                'mean': np.mean(preemption_gaps) * gap_seconds / 3600,  # hours
                'median': np.median(preemption_gaps) * gap_seconds / 3600,
                'std': np.std(preemption_gaps) * gap_seconds / 3600,
                'min': min(preemption_gaps) * gap_seconds / 3600,
                'max': max(preemption_gaps) * gap_seconds / 3600,
                'count': len(preemption_gaps)
            }
        else:
            gap_stats = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
            }
        
        # Calculate additional metrics
        total_hours = len(data) * gap_seconds / 3600
        uptime_hours = sum(availability_data) * gap_seconds / 3600
        
        all_trace_stats.append({
            'trace_id': i,
            'availability_rate': availability_rate,
            'total_hours': total_hours,
            'uptime_hours': uptime_hours,
            'spot_duration_mean_hours': spot_duration_hours['mean'],
            'spot_duration_median_hours': spot_duration_hours['median'],
            'spot_duration_std_hours': spot_duration_hours['std'],
            'spot_duration_max_hours': spot_duration_hours['max'],
            'spot_duration_count': spot_duration_hours['count'],
            'preemption_gap_mean_hours': gap_stats['mean'],
            'preemption_gap_median_hours': gap_stats['median'],
            'preemption_gap_std_hours': gap_stats['std'],
            'preemption_gap_max_hours': gap_stats['max'],
            'preemption_gap_count': gap_stats['count'],
            'gap_seconds': gap_seconds,
            'raw_spot_durations': spot_durations,
            'raw_preemption_gaps': preemption_gaps
        })
    
    if not all_trace_stats:
        return {}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_trace_stats)
    
    # Calculate correlations
    correlations = {}
    correlation_fields = [
        'spot_duration_mean_hours', 'spot_duration_median_hours', 'spot_duration_max_hours',
        'preemption_gap_mean_hours', 'preemption_gap_median_hours', 'preemption_gap_max_hours'
    ]
    
    for field in correlation_fields:
        if df[field].std() > 0:  # Avoid division by zero
            pearson_r, pearson_p = pearsonr(df['availability_rate'], df[field])
            spearman_r, spearman_p = spearmanr(df['availability_rate'], df[field])
            correlations[field] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
        else:
            correlations[field] = {
                'pearson_r': 0, 'pearson_p': 1, 'spearman_r': 0, 'spearman_p': 1
            }
    
    # Aggregate statistics
    summary_stats = {
        'region': region,
        'num_traces': len(all_trace_stats),
        'availability_rate_mean': df['availability_rate'].mean(),
        'availability_rate_std': df['availability_rate'].std(),
        'availability_rate_min': df['availability_rate'].min(),
        'availability_rate_max': df['availability_rate'].max(),
        
        'spot_duration_mean_hours': df['spot_duration_mean_hours'].mean(),
        'spot_duration_median_hours': df['spot_duration_median_hours'].mean(),
        'spot_duration_max_hours': df['spot_duration_max_hours'].mean(),
        'spot_duration_mean_std': df['spot_duration_mean_hours'].std(),
        
        'preemption_gap_mean_hours': df['preemption_gap_mean_hours'].mean(),
        'preemption_gap_median_hours': df['preemption_gap_median_hours'].mean(),
        'preemption_gap_max_hours': df['preemption_gap_max_hours'].mean(),
        'preemption_gap_mean_std': df['preemption_gap_mean_hours'].std(),
        
        'correlations': correlations,
        'trace_data': df
    }
    
    return summary_stats

def create_correlation_analysis_plots(all_results: List[Dict], output_dir: Path):
    """Create focused correlation analysis plot."""
    # Filter out regions with no data
    valid_results = [r for r in all_results if r]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Create focused single plot
    plt.figure(figsize=(12, 8))
    
    # Collect data across all regions
    all_avail_rates = []
    all_spot_durations = []
    region_labels = []
    colors = []
    
    # Define region colors for better visualization
    region_colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
    
    for i, result in enumerate(valid_results):
        df = result['trace_data']
        region = result['region'].replace('_v100_1', '')
        
        avail_rates = df['availability_rate'].tolist()
        spot_durations = df['spot_duration_mean_hours'].tolist()
        
        all_avail_rates.extend(avail_rates)
        all_spot_durations.extend(spot_durations)
        region_labels.extend([region] * len(df))
        colors.extend([region_colors[i]] * len(df))
    
    # Create scatter plot with region colors
    for i, result in enumerate(valid_results):
        df = result['trace_data']
        region = result['region'].replace('_v100_1', '')
        plt.scatter(df['availability_rate'], df['spot_duration_mean_hours'], 
                   alpha=0.6, s=30, color=region_colors[i], label=region)
    
    # Add overall trend line
    if len(all_avail_rates) > 1 and np.std(all_spot_durations) > 0:
        z = np.polyfit(all_avail_rates, all_spot_durations, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(all_avail_rates), max(all_avail_rates), 100)
        plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Overall Trend')
        
        # Calculate correlation
        r, p_val = pearsonr(all_avail_rates, all_spot_durations)
        
        # Add correlation info box
        textstr = f'Overall Correlation:\nr = {r:.3f}\np = {p_val:.3e}\n\n'
        if abs(r) > 0.5:
            strength = "Strong"
        elif abs(r) > 0.3:
            strength = "Moderate" 
        else:
            strength = "Weak"
        
        direction = "positive" if r > 0 else "negative"
        textstr += f'{strength} {direction} correlation'
        
        # Add interpretation
        if r > 0.3 and p_val < 0.05:
            textstr += '\n\n✓ Higher availability regions\n  tend to have longer\n  spot durations'
        
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.8), fontsize=11)
    
    plt.xlabel('Availability Rate', fontsize=14)
    plt.ylabel('Mean Spot Duration (hours)', fontsize=14)
    plt.title('Availability Rate vs Spot Duration Relationship\n(Each point = one trace sample)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add summary stats in corner
    summary_text = f'Total: {len(all_avail_rates)} samples\nRegions: {len(valid_results)}'
    plt.text(0.98, 0.02, summary_text, transform=plt.gca().transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'availability_duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(all_results: List[Dict], output_path: Path):
    """Generate detailed analysis report."""
    valid_results = [r for r in all_results if r]
    
    with open(output_path, 'w') as f:
        f.write("# Availability Rate vs Spot Duration Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        
        if not valid_results:
            f.write("No valid data found for analysis.\n")
            return
        
        # Calculate overall correlations
        all_avail_rates = []
        all_spot_durations = []
        all_gap_durations = []
        
        for result in valid_results:
            df = result['trace_data']
            all_avail_rates.extend(df['availability_rate'].tolist())
            all_spot_durations.extend(df['spot_duration_mean_hours'].tolist())
            all_gap_durations.extend(df['preemption_gap_mean_hours'].tolist())
        
        if len(all_avail_rates) > 1:
            spot_corr, spot_p = pearsonr(all_avail_rates, all_spot_durations)
            gap_corr, gap_p = pearsonr(all_avail_rates, all_gap_durations)
            
            f.write(f"- **Total regions analyzed**: {len(valid_results)}\n")
            f.write(f"- **Total trace samples**: {sum(r['num_traces'] for r in valid_results)}\n")
            f.write(f"- **Overall availability-spot duration correlation**: r = {spot_corr:.3f} (p = {spot_p:.3f})\n")
            f.write(f"- **Overall availability-gap duration correlation**: r = {gap_corr:.3f} (p = {gap_p:.3f})\n\n")
            
            # Interpret correlations
            f.write("### Key Findings\n\n")
            
            if abs(spot_corr) > 0.3 and spot_p < 0.05:
                direction = "longer" if spot_corr > 0 else "shorter"
                f.write(f"🔍 **Significant relationship found**: Regions with higher availability rates tend to have {direction} spot instance durations.\n")
            else:
                f.write("📊 **No strong relationship**: No significant correlation between availability rate and spot duration.\n")
            
            if abs(gap_corr) > 0.3 and gap_p < 0.05:
                direction = "longer" if gap_corr > 0 else "shorter"
                f.write(f"🔍 **Preemption pattern**: Regions with higher availability rates tend to have {direction} preemption gaps.\n")
            else:
                f.write("📊 **Preemption gaps**: No significant correlation between availability rate and preemption gap duration.\n")
        
        f.write("\n## Detailed Analysis by Region\n\n")
        
        # Sort regions by availability rate for easier comparison
        sorted_results = sorted(valid_results, key=lambda x: x['availability_rate_mean'], reverse=True)
        
        for result in sorted_results:
            region = result['region']
            f.write(f"### {region}\n\n")
            
            f.write("#### Summary Statistics\n\n")
            f.write("| Metric | Mean | Std Dev | Min | Max |\n")
            f.write("|--------|------|---------|-----|-----|\n")
            f.write(f"| Availability Rate | {result['availability_rate_mean']:.3f} | "
                   f"{result['availability_rate_std']:.3f} | "
                   f"{result['availability_rate_min']:.3f} | "
                   f"{result['availability_rate_max']:.3f} |\n")
            f.write(f"| Spot Duration (hours) | {result['spot_duration_mean_hours']:.2f} | "
                   f"{result['spot_duration_mean_std']:.2f} | - | "
                   f"{result['spot_duration_max_hours']:.2f} |\n")
            f.write(f"| Preemption Gap (hours) | {result['preemption_gap_mean_hours']:.2f} | "
                   f"{result['preemption_gap_mean_std']:.2f} | - | "
                   f"{result['preemption_gap_max_hours']:.2f} |\n")
            
            f.write("\n#### Correlation Analysis\n\n")
            f.write("| Duration Metric | Pearson r | p-value | Interpretation |\n")
            f.write("|----------------|-----------|---------|----------------|\n")
            
            for metric, corr_data in result['correlations'].items():
                r = corr_data['pearson_r']
                p = corr_data['pearson_p']
                
                if abs(r) > 0.5:
                    strength = "Strong"
                elif abs(r) > 0.3:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                direction = "positive" if r > 0 else "negative"
                significance = "significant" if p < 0.05 else "not significant"
                
                f.write(f"| {metric.replace('_', ' ').title()} | {r:.3f} | {p:.3f} | "
                       f"{strength} {direction} ({significance}) |\n")
            
            f.write("\n---\n\n")
        
        f.write("## Methodology\n\n")
        f.write("1. **Data Source**: Analyzed aligned traces from `data/converted_multi_region_aligned/`\n")
        f.write("2. **Metrics Calculated**:\n")
        f.write("   - Availability rate: Proportion of time instances are available\n")
        f.write("   - Spot duration: Length of continuous available periods\n")
        f.write("   - Preemption gap: Length of continuous unavailable periods\n")
        f.write("3. **Statistical Analysis**:\n")
        f.write("   - Pearson correlation for linear relationships\n")
        f.write("   - Spearman correlation for monotonic relationships\n")
        f.write("4. **Data Format**: Traces use preempted format (0=available, 1=preempted)\n")
        
        f.write("\n## Conclusions\n\n")
        
        # Generate conclusions based on findings
        high_avail_regions = [r for r in valid_results if r['availability_rate_mean'] > 0.8]
        low_avail_regions = [r for r in valid_results if r['availability_rate_mean'] < 0.5]
        
        if high_avail_regions and low_avail_regions:
            high_avg_duration = np.mean([r['spot_duration_mean_hours'] for r in high_avail_regions])
            low_avg_duration = np.mean([r['spot_duration_mean_hours'] for r in low_avail_regions])
            
            f.write(f"**Regional Comparison:**\n")
            f.write(f"- High availability regions (>80%): avg duration = {high_avg_duration:.2f} hours\n")
            f.write(f"- Low availability regions (<50%): avg duration = {low_avg_duration:.2f} hours\n")
            
            if high_avg_duration > low_avg_duration * 1.2:
                f.write(f"- **Pattern identified**: Good regions tend to have longer spot durations\n")
            elif low_avg_duration > high_avg_duration * 1.2:
                f.write(f"- **Pattern identified**: Poor regions tend to have longer spot durations\n")
            else:
                f.write(f"- **No clear pattern**: Duration differences between good and poor regions are minimal\n")
        
        f.write(f"\nThis analysis helps understand whether spot instance scheduling should consider not just availability rates, ")
        f.write(f"but also typical duration patterns when choosing between regions.\n")

def main():
    """Run availability-duration relationship analysis."""
    # Get all regions from aligned trace data
    trace_base_dir = Path("data/converted_multi_region_aligned")
    all_regions = sorted([d.name for d in trace_base_dir.iterdir() if d.is_dir() and d.name.endswith('_v100_1')])
    
    print("="*80)
    print("AVAILABILITY RATE vs SPOT DURATION ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(all_regions)} regions...\n")
    
    results = []
    
    for region in all_regions:
        print(f"Analyzing {region}...", end='', flush=True)
        
        region_stats = analyze_availability_and_durations(region)
        
        if region_stats:
            results.append(region_stats)
            print(f" ✓ ({region_stats['num_traces']} traces)")
        else:
            print(" ✗ (no data)")
    
    if not results:
        print("No valid data found. Exiting.")
        return
    
    # Create output directory
    output_dir = Path("outputs/availability_duration_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating analysis plots...")
    create_correlation_analysis_plots(results, output_dir)
    
    # Generate report
    print("Generating detailed report...")
    generate_detailed_report(results, output_dir / "availability_duration_report.md")
    
    # Quick summary for console
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - Report: {output_dir}/availability_duration_report.md")
    print(f"   - Plots: {output_dir}/availability_duration_analysis.png")
    
    # Print quick summary
    valid_results = [r for r in results if r]
    if valid_results:
        all_avail_rates = []
        all_spot_durations = []
        
        for result in valid_results:
            df = result['trace_data']
            all_avail_rates.extend(df['availability_rate'].tolist())
            all_spot_durations.extend(df['spot_duration_mean_hours'].tolist())
        
        if len(all_avail_rates) > 1:
            from scipy.stats import pearsonr
            corr, p_val = pearsonr(all_avail_rates, all_spot_durations)
            print(f"\n📊 Quick Summary:")
            print(f"   - Overall correlation (availability vs duration): r = {corr:.3f}, p = {p_val:.3f}")
            if abs(corr) > 0.3 and p_val < 0.05:
                direction = "longer" if corr > 0 else "shorter"
                print(f"   - 🔍 Finding: Higher availability → {direction} spot durations")
            else:
                print(f"   - 📈 Finding: No significant relationship between availability and duration")

if __name__ == "__main__":
    main()