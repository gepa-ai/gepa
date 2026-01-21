#!/usr/bin/env python3
"""
Analyze sampling consistency for all regions.
Compare full CSV data with sampled aligned traces.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def analyze_full_csv(csv_path: Path) -> Dict:
    """Analyze full CSV file statistics."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    availability_rate = df['availability'].mean()
    
    # Calculate run lengths
    runs = []
    if len(df) > 0:
        current_value = df['availability'].iloc[0]
        current_length = 1
        
        for i in range(1, len(df)):
            if df['availability'].iloc[i] == current_value:
                current_length += 1
            else:
                runs.append((current_value, current_length))
                current_value = df['availability'].iloc[i]
                current_length = 1
        runs.append((current_value, current_length))
    
    available_runs = [length for value, length in runs if value == 1]
    unavailable_runs = [length for value, length in runs if value == 0]
    
    # Calculate autocorrelation
    autocorr_lag1 = df['availability'].autocorr(lag=1) if len(df) > 1 else 0
    autocorr_lag10 = df['availability'].autocorr(lag=10) if len(df) > 10 else 0
    
    return {
        'total_points': len(df),
        'availability_rate': availability_rate,
        'availability_std': df['availability'].std(),
        'available_run_mean': np.mean(available_runs) if available_runs else 0,
        'available_run_median': np.median(available_runs) if available_runs else 0,
        'available_run_max': max(available_runs) if available_runs else 0,
        'unavailable_run_mean': np.mean(unavailable_runs) if unavailable_runs else 0,
        'unavailable_run_median': np.median(unavailable_runs) if unavailable_runs else 0,
        'unavailable_run_max': max(unavailable_runs) if unavailable_runs else 0,
        'autocorr_lag1': autocorr_lag1,
        'autocorr_lag10': autocorr_lag10,
        'gap_seconds': int((df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()) if len(df) > 1 else 195
    }

def analyze_aligned_traces(region: str, num_traces: int = 100) -> Dict:
    """Analyze aligned trace segments."""
    trace_dir = Path(f"../../data/converted_multi_region_aligned/{region}")
    
    if not trace_dir.exists():
        return {}
    
    all_stats = []
    
    for i in range(min(num_traces, len(list(trace_dir.glob("*.json"))))):
        with open(trace_dir / f"{i}.json", 'r') as f:
            trace = json.load(f)
        
        data = trace['data']
        # Convert from preempted format (0=available, 1=preempted) to availability
        availability_data = [1 - x for x in data]
        
        availability_rate = sum(availability_data) / len(availability_data) if availability_data else 0
        
        # Calculate runs
        runs = []
        if data:
            current_value = availability_data[0]
            current_length = 1
            
            for j in range(1, len(availability_data)):
                if availability_data[j] == current_value:
                    current_length += 1
                else:
                    runs.append((current_value, current_length))
                    current_value = availability_data[j]
                    current_length = 1
            runs.append((current_value, current_length))
        
        available_runs = [length for value, length in runs if value == 1]
        unavailable_runs = [length for value, length in runs if value == 0]
        
        # Calculate autocorrelation
        avail_series = pd.Series(availability_data)
        autocorr_lag1 = avail_series.autocorr(lag=1) if len(avail_series) > 1 else 0
        autocorr_lag10 = avail_series.autocorr(lag=10) if len(avail_series) > 10 else 0
        
        all_stats.append({
            'trace_id': i,
            'availability_rate': availability_rate,
            'available_run_mean': np.mean(available_runs) if available_runs else 0,
            'available_run_median': np.median(available_runs) if available_runs else 0,
            'available_run_max': max(available_runs) if available_runs else 0,
            'unavailable_run_mean': np.mean(unavailable_runs) if unavailable_runs else 0,
            'unavailable_run_median': np.median(unavailable_runs) if unavailable_runs else 0,
            'unavailable_run_max': max(unavailable_runs) if unavailable_runs else 0,
            'autocorr_lag1': autocorr_lag1,
            'autocorr_lag10': autocorr_lag10,
            'trace_length': len(data)
        })
    
    # Aggregate statistics
    df_stats = pd.DataFrame(all_stats)
    
    return {
        'num_traces': len(all_stats),
        'trace_length_mean': df_stats['trace_length'].mean(),
        'availability_rate_mean': df_stats['availability_rate'].mean(),
        'availability_rate_std': df_stats['availability_rate'].std(),
        'availability_rate_min': df_stats['availability_rate'].min(),
        'availability_rate_max': df_stats['availability_rate'].max(),
        'available_run_mean': df_stats['available_run_mean'].mean(),
        'available_run_median': df_stats['available_run_median'].mean(),
        'unavailable_run_mean': df_stats['unavailable_run_mean'].mean(),
        'unavailable_run_median': df_stats['unavailable_run_median'].mean(),
        'autocorr_lag1_mean': df_stats['autocorr_lag1'].mean(),
        'autocorr_lag10_mean': df_stats['autocorr_lag10'].mean(),
        'all_traces_stats': df_stats
    }

def create_comparison_plots(results: List[Dict], output_dir: Path):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(20, 12))
    
    # Filter out regions with no data
    results = [r for r in results if r['aligned']]
    regions = [r['region'].replace('_v100_1', '') for r in results]
    
    # 1. Availability Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(regions))
    width = 0.35
    
    full_rates = [r['full']['availability_rate'] for r in results]
    aligned_rates = [r['aligned']['availability_rate_mean'] for r in results]
    aligned_std = [r['aligned']['availability_rate_std'] for r in results]
    
    ax1.bar(x - width/2, full_rates, width, label='Full CSV', alpha=0.8)
    ax1.bar(x + width/2, aligned_rates, width, yerr=aligned_std, 
            label='Aligned Traces (mean±std)', alpha=0.8, capsize=5)
    ax1.set_ylabel('Availability Rate')
    ax1.set_title('Availability Rate Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Absolute Difference
    ax2 = plt.subplot(2, 3, 2)
    diffs = [abs(r['full']['availability_rate'] - r['aligned']['availability_rate_mean']) for r in results]
    colors = ['green' if d < 0.05 else 'orange' if d < 0.1 else 'red' for d in diffs]
    ax2.bar(x, diffs, color=colors, alpha=0.7)
    ax2.set_ylabel('Absolute Difference')
    ax2.set_title('Sampling Accuracy (|Full - Aligned Mean|)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45)
    ax2.axhline(y=0.05, color='orange', linestyle='--', label='5% threshold')
    ax2.axhline(y=0.1, color='red', linestyle='--', label='10% threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Available Run Length Comparison
    ax3 = plt.subplot(2, 3, 3)
    full_avail_runs = [r['full']['available_run_mean'] for r in results]
    aligned_avail_runs = [r['aligned']['available_run_mean'] for r in results]
    
    ax3.bar(x - width/2, full_avail_runs, width, label='Full CSV', alpha=0.8, color='green')
    ax3.bar(x + width/2, aligned_avail_runs, width, label='Aligned Traces', alpha=0.8, color='lightgreen')
    ax3.set_ylabel('Mean Run Length (ticks)')
    ax3.set_title('Mean Available Run Length')
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Unavailable Run Length Comparison
    ax4 = plt.subplot(2, 3, 4)
    full_unavail_runs = [r['full']['unavailable_run_mean'] for r in results]
    aligned_unavail_runs = [r['aligned']['unavailable_run_mean'] for r in results]
    
    ax4.bar(x - width/2, full_unavail_runs, width, label='Full CSV', alpha=0.8, color='red')
    ax4.bar(x + width/2, aligned_unavail_runs, width, label='Aligned Traces', alpha=0.8, color='pink')
    ax4.set_ylabel('Mean Run Length (ticks)')
    ax4.set_title('Mean Unavailable Run Length')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regions, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Autocorrelation Comparison
    ax5 = plt.subplot(2, 3, 5)
    full_autocorr = [r['full']['autocorr_lag1'] for r in results]
    aligned_autocorr = [r['aligned']['autocorr_lag1_mean'] for r in results]
    
    ax5.bar(x - width/2, full_autocorr, width, label='Full CSV', alpha=0.8)
    ax5.bar(x + width/2, aligned_autocorr, width, label='Aligned Traces', alpha=0.8)
    ax5.set_ylabel('Autocorrelation')
    ax5.set_title('Lag-1 Autocorrelation Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(regions, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Availability Rate Distribution (for selected region)
    ax6 = plt.subplot(2, 3, 6)
    # Pick region with most variation
    region_idx = np.argmax([r['aligned']['availability_rate_std'] for r in results])
    region_data = results[region_idx]
    
    if 'all_traces_stats' in region_data['aligned']:
        trace_rates = region_data['aligned']['all_traces_stats']['availability_rate']
        ax6.hist(trace_rates, bins=20, alpha=0.7, edgecolor='black')
        ax6.axvline(region_data['full']['availability_rate'], color='red', 
                   linestyle='--', linewidth=2, label='Full CSV rate')
        ax6.axvline(region_data['aligned']['availability_rate_mean'], color='blue', 
                   linestyle='--', linewidth=2, label='Aligned mean')
        ax6.set_xlabel('Availability Rate')
        ax6.set_ylabel('Number of Traces')
        ax6.set_title(f'Availability Rate Distribution\n({regions[region_idx]})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Trace Sampling Consistency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'sampling_consistency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(results: List[Dict], output_path: Path):
    """Generate detailed analysis report."""
    with open(output_path, 'w') as f:
        f.write("# Trace Sampling Consistency Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        
        # Overall statistics
        valid_results = [r for r in results if r['aligned']]
        all_diffs = [abs(r['full']['availability_rate'] - r['aligned']['availability_rate_mean']) 
                    for r in valid_results]
        
        f.write(f"- **Total regions analyzed**: {len(valid_results)}\n")
        f.write(f"- **Mean availability rate difference**: {np.mean(all_diffs):.3f}\n")
        f.write(f"- **Max availability rate difference**: {np.max(all_diffs):.3f}\n")
        f.write(f"- **Regions within 5% difference**: {sum(1 for d in all_diffs if d < 0.05)}/{len(all_diffs)}\n")
        f.write(f"- **Regions within 10% difference**: {sum(1 for d in all_diffs if d < 0.1)}/{len(all_diffs)}\n\n")
        
        f.write("## Detailed Analysis by Region\n\n")
        
        for result in results:
            region = result['region']
            f.write(f"### {region}\n\n")
            
            if not result['aligned']:
                f.write("No aligned trace data found for this region.\n\n")
                continue
            
            full = result['full']
            aligned = result['aligned']
            diff = result['diff']
            
            f.write("#### Basic Statistics\n\n")
            f.write("| Metric | Full CSV | Aligned Traces | Difference |\n")
            f.write("|--------|----------|----------------|------------|\n")
            f.write(f"| Availability Rate | {full['availability_rate']:.3f} | "
                   f"{aligned['availability_rate_mean']:.3f} ± {aligned['availability_rate_std']:.3f} | "
                   f"{diff:.3f} |\n")
            f.write(f"| Data Points | {full['total_points']:,} | "
                   f"{int(aligned['num_traces'] * aligned['trace_length_mean']):,} | "
                   f"{(aligned['num_traces'] * aligned['trace_length_mean'] / full['total_points']):.1%} sampled |\n")
            
            f.write("\n#### Run Length Statistics\n\n")
            f.write("| Run Type | Metric | Full CSV | Aligned Traces |\n")
            f.write("|----------|--------|----------|----------------|\n")
            f.write(f"| Available | Mean | {full['available_run_mean']:.1f} | {aligned['available_run_mean']:.1f} |\n")
            f.write(f"| Available | Median | {full['available_run_median']:.1f} | {aligned['available_run_median']:.1f} |\n")
            f.write(f"| Available | Max | {full['available_run_max']} | - |\n")
            f.write(f"| Unavailable | Mean | {full['unavailable_run_mean']:.1f} | {aligned['unavailable_run_mean']:.1f} |\n")
            f.write(f"| Unavailable | Median | {full['unavailable_run_median']:.1f} | {aligned['unavailable_run_median']:.1f} |\n")
            f.write(f"| Unavailable | Max | {full['unavailable_run_max']} | - |\n")
            
            f.write("\n#### Temporal Correlation\n\n")
            f.write(f"- **Lag-1 Autocorrelation**: Full CSV = {full['autocorr_lag1']:.3f}, "
                   f"Aligned = {aligned['autocorr_lag1_mean']:.3f}\n")
            f.write(f"- **Lag-10 Autocorrelation**: Full CSV = {full['autocorr_lag10']:.3f}, "
                   f"Aligned = {aligned['autocorr_lag10_mean']:.3f}\n")
            
            # Assessment
            f.write("\n#### Assessment\n\n")
            if diff < 0.05:
                f.write("✅ **Excellent**: Sampling preserves availability characteristics very well (< 5% difference)\n")
            elif diff < 0.1:
                f.write("⚠️ **Good**: Sampling preserves availability characteristics reasonably well (< 10% difference)\n")
            else:
                f.write("❌ **Poor**: Significant sampling bias detected (> 10% difference)\n")
            
            f.write("\n---\n\n")
        
        f.write("## Methodology\n\n")
        f.write("1. **Full CSV Analysis**: Analyzed complete availability data from `data/real/availability/2023-02-15/processed/`\n")
        f.write("2. **Aligned Trace Analysis**: Analyzed 100 aligned traces from `data/converted_multi_region_aligned/`\n")
        f.write("3. **Metrics Compared**:\n")
        f.write("   - Availability rate (mean and distribution)\n")
        f.write("   - Run length statistics (continuous available/unavailable periods)\n")
        f.write("   - Temporal autocorrelation (persistence of states)\n")
        f.write("4. **Data Format Note**: Aligned traces use preempted format (0=available, 1=preempted) matching TraceEnv expectations\n")
        
        f.write("\n## Conclusions\n\n")
        
        if np.mean(all_diffs) < 0.05:
            f.write("The trace sampling method demonstrates **excellent consistency** with the original data. ")
            f.write("The aligned traces accurately preserve the statistical properties of the full traces, ")
            f.write("making them suitable for multi-region strategy evaluation.\n")
        elif np.mean(all_diffs) < 0.1:
            f.write("The trace sampling method demonstrates **good consistency** with the original data. ")
            f.write("While there are some minor differences, the aligned traces generally preserve ")
            f.write("the key statistical properties needed for strategy evaluation.\n")
        else:
            f.write("The trace sampling method shows **significant bias** compared to the original data. ")
            f.write("Care should be taken when interpreting results, as the sampling may not accurately ")
            f.write("represent the true availability patterns.\n")

def main():
    """Run comprehensive sampling consistency analysis."""
    # Get all regions from CSV data
    csv_dir = Path("../../data/real/availability/2023-02-15/processed")
    all_regions = sorted([f.stem for f in csv_dir.glob("*_v100_1.csv")])
    
    print("="*80)
    print("TRACE SAMPLING CONSISTENCY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(all_regions)} regions...\n")
    
    results = []
    
    for region in all_regions:
        print(f"Analyzing {region}...", end='', flush=True)
        
        csv_path = csv_dir / f"{region}.csv"
        full_stats = analyze_full_csv(csv_path)
        aligned_stats = analyze_aligned_traces(region)
        
        if aligned_stats:
            diff = abs(full_stats['availability_rate'] - aligned_stats['availability_rate_mean'])
            results.append({
                'region': region,
                'full': full_stats,
                'aligned': aligned_stats,
                'diff': diff
            })
            print(f" ✓ (diff: {diff:.3f})")
        else:
            results.append({
                'region': region,
                'full': full_stats,
                'aligned': None,
                'diff': None
            })
            print(" ✗ (no aligned traces)")
    
    # Create output directory
    output_dir = Path("../../outputs/trace_sampling_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(results, output_dir)
    
    # Generate report
    print("Generating detailed report...")
    generate_report(results, output_dir / "sampling_consistency_report.md")
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"   - Report: {output_dir}/sampling_consistency_report.md")
    print(f"   - Plots: {output_dir}/sampling_consistency_analysis.png")

if __name__ == "__main__":
    main()