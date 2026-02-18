#!/usr/bin/env python3
"""
Analyze spot duration consistency within sliding time windows.
Tests if recent spot durations can predict immediate next durations.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

def analyze_sliding_window_prediction(region: str, num_traces: int = 50) -> dict:
    """Analyze prediction accuracy within sliding time windows."""
    trace_dir = Path(f"data/converted_multi_region_aligned/{region}")
    
    if not trace_dir.exists():
        return {}
    
    # Different window sizes to test (in hours) - extend to larger windows
    window_sizes = [1, 2, 4, 6, 12, 24, 48, 72]
    results = {ws: [] for ws in window_sizes}
    
    for i in range(min(num_traces, len(list(trace_dir.glob("*.json"))))):
        trace_path = trace_dir / f"{i}.json"
        if not trace_path.exists():
            continue
            
        with open(trace_path, 'r') as f:
            trace = json.load(f)
        
        data = trace['data']
        gap_seconds = trace['metadata']['gap_seconds']
        
        # Convert to availability timeline with timestamps
        availability_timeline = []
        current_time = 0
        
        for available in [1 - x for x in data]:  # Convert from preempted format
            availability_timeline.append((current_time, available))
            current_time += gap_seconds / 3600  # Convert to hours
        
        # Extract spot instances with start/end times
        spot_instances = []
        current_start = None
        
        for time_hour, available in availability_timeline:
            if available == 1 and current_start is None:
                current_start = time_hour
            elif available == 0 and current_start is not None:
                duration = time_hour - current_start
                spot_instances.append({
                    'start': current_start,
                    'end': time_hour,
                    'duration': duration
                })
                current_start = None
        
        # Handle case where trace ends with availability
        if current_start is not None:
            duration = availability_timeline[-1][0] - current_start
            spot_instances.append({
                'start': current_start,
                'end': availability_timeline[-1][0],
                'duration': duration
            })
        
        # Skip if too few instances
        if len(spot_instances) < 3:
            continue
        
        # Test each window size
        for window_hours in window_sizes:
            window_predictions = []
            
            # For each spot instance (except the first), try to predict its duration
            for target_idx in range(1, len(spot_instances)):
                target_spot = spot_instances[target_idx]
                target_start = target_spot['start']
                
                # Find all spots that ended within the window before target_start
                window_start = target_start - window_hours
                recent_spots = []
                
                for j in range(target_idx):
                    prev_spot = spot_instances[j]
                    if prev_spot['end'] >= window_start and prev_spot['end'] <= target_start:
                        recent_spots.append(prev_spot)
                
                # Skip if no recent spots in window
                if not recent_spots:
                    continue
                
                # Calculate average duration of recent spots (this is our prediction)
                predicted_duration = np.mean([s['duration'] for s in recent_spots])
                actual_duration = target_spot['duration']
                
                # Calculate prediction error metrics
                absolute_error = abs(predicted_duration - actual_duration)
                relative_error = absolute_error / actual_duration if actual_duration > 0 else 0
                squared_error = (predicted_duration - actual_duration) ** 2
                
                window_predictions.append({
                    'predicted_duration': predicted_duration,
                    'actual_duration': actual_duration,
                    'absolute_error': absolute_error,
                    'relative_error': relative_error,
                    'squared_error': squared_error,
                    'num_recent_spots': len(recent_spots)
                })
            
            results[window_hours].extend(window_predictions)
    
    # Calculate summary statistics for each window size
    summary = {}
    for window_hours in window_sizes:
        predictions = results[window_hours]
        if not predictions:
            continue
        
        total_predictions = len(predictions)
        predicted_durations = [p['predicted_duration'] for p in predictions]
        actual_durations = [p['actual_duration'] for p in predictions]
        
        # Calculate regression metrics
        mae = np.mean([p['absolute_error'] for p in predictions])  # Mean Absolute Error
        rmse = np.sqrt(np.mean([p['squared_error'] for p in predictions]))  # Root Mean Square Error
        mape = np.mean([p['relative_error'] for p in predictions]) * 100  # Mean Absolute Percentage Error
        
        # Calculate correlation coefficient
        if len(predicted_durations) > 1 and np.std(predicted_durations) > 0 and np.std(actual_durations) > 0:
            correlation = np.corrcoef(predicted_durations, actual_durations)[0, 1]
        else:
            correlation = 0
            
        # Calculate R-squared
        if np.var(actual_durations) > 0:
            ss_res = np.sum([(p['predicted_duration'] - p['actual_duration'])**2 for p in predictions])
            ss_tot = np.sum([(d - np.mean(actual_durations))**2 for d in actual_durations])
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0
        
        summary[window_hours] = {
            'total_predictions': total_predictions,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'r_squared': r_squared,
            'predictions': predictions
        }
    
    return {
        'region': region,
        'window_results': summary
    }

def create_sliding_window_plot(all_results: List[dict], output_dir: Path):
    """Create visualization of sliding window prediction results."""
    plt.figure(figsize=(18, 10))
    
    # Collect data across all regions - include extended window sizes
    window_sizes = [1, 2, 4, 6, 12, 24, 48, 72]
    region_names = []
    correlations_by_window = {ws: [] for ws in window_sizes}
    r_squared_by_window = {ws: [] for ws in window_sizes}
    mae_by_window = {ws: [] for ws in window_sizes}
    
    for result in all_results:
        if not result or 'window_results' not in result:
            continue
        
        region_names.append(result['region'].replace('_v100_1', ''))
        
        for ws in window_sizes:
            if ws in result['window_results']:
                correlations_by_window[ws].append(result['window_results'][ws]['correlation'])
                r_squared_by_window[ws].append(result['window_results'][ws]['r_squared'])
                mae_by_window[ws].append(result['window_results'][ws]['mae'])
            else:
                correlations_by_window[ws].append(0)
                r_squared_by_window[ws].append(0)
                mae_by_window[ws].append(1.0)  # Poor MAE as default
    
    # Plot 1: Correlation by window size
    plt.subplot(2, 2, 1)
    
    avg_correlations = []
    std_correlations = []
    
    for ws in window_sizes:
        if correlations_by_window[ws]:
            avg_corr = np.mean(correlations_by_window[ws])
            std_corr = np.std(correlations_by_window[ws])
        else:
            avg_corr = 0
            std_corr = 0
        
        avg_correlations.append(avg_corr)
        std_correlations.append(std_corr)
    
    plt.errorbar(window_sizes, avg_correlations, yerr=std_correlations, 
                marker='o', capsize=5, linewidth=2, markersize=8, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Correlation')
    plt.xlabel('Window Size (hours)', fontweight='bold')
    plt.ylabel('Correlation Coefficient', fontweight='bold')
    plt.title('Prediction Correlation by Window Size', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.0)
    
    # Plot 2: R-squared by window size
    plt.subplot(2, 2, 2)
    
    avg_r_squared = []
    std_r_squared = []
    
    for ws in window_sizes:
        if r_squared_by_window[ws]:
            avg_r2 = np.mean(r_squared_by_window[ws])
            std_r2 = np.std(r_squared_by_window[ws])
        else:
            avg_r2 = 0
            std_r2 = 0
        
        avg_r_squared.append(avg_r2)
        std_r_squared.append(std_r2)
    
    plt.errorbar(window_sizes, avg_r_squared, yerr=std_r_squared, 
                marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No Predictive Power')
    plt.xlabel('Window Size (hours)', fontweight='bold')
    plt.ylabel('R-squared', fontweight='bold')
    plt.title('Prediction R² by Window Size', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.0)
    
    # Plot 3: MAE by window size
    plt.subplot(2, 2, 3)
    
    avg_mae = []
    std_mae = []
    
    for ws in window_sizes:
        if mae_by_window[ws]:
            avg_error = np.mean(mae_by_window[ws])
            std_error = np.std(mae_by_window[ws])
        else:
            avg_error = 1.0
            std_error = 0
        
        avg_mae.append(avg_error)
        std_mae.append(std_error)
    
    plt.errorbar(window_sizes, avg_mae, yerr=std_mae, 
                marker='^', capsize=5, linewidth=2, markersize=8, color='orange')
    plt.xlabel('Window Size (hours)', fontweight='bold')
    plt.ylabel('Mean Absolute Error (hours)', fontweight='bold')
    plt.title('Prediction Error by Window Size', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Sample count distribution
    plt.subplot(2, 2, 4)
    
    sample_counts = []
    for ws in window_sizes:
        total_samples = sum(len(result['window_results'].get(ws, {}).get('predictions', [])) 
                          for result in all_results 
                          if result and 'window_results' in result)
        sample_counts.append(total_samples)
    
    bars = plt.bar(range(len(window_sizes)), sample_counts, alpha=0.7, color='purple')
    plt.xlabel('Window Size (hours)', fontweight='bold')
    plt.ylabel('Total Predictions', fontweight='bold')
    plt.title('Number of Predictions by Window Size', fontweight='bold')
    plt.xticks(range(len(window_sizes)), [f'{ws}h' for ws in window_sizes])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(sample_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Sliding Window Spot Duration Prediction Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'sliding_window_prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run sliding window prediction analysis."""
    # Get all regions
    trace_base_dir = Path("data/converted_multi_region_aligned")
    all_regions = sorted([d.name for d in trace_base_dir.iterdir() if d.is_dir() and d.name.endswith('_v100_1')])
    
    print("="*80)
    print("SLIDING WINDOW SPOT DURATION PREDICTION ANALYSIS (FIXED)")
    print("="*80)
    print(f"\nTesting if recent spot durations can predict the next one...")
    print(f"Window sizes: 1h, 2h, 4h, 6h, 12h, 24h, 48h, 72h")
    print(f"Now using continuous prediction (not binary classification)\n")
    
    results = []
    
    for region in all_regions:  # Use all regions
        print(f"Analyzing {region}...", end='', flush=True)
        
        region_stats = analyze_sliding_window_prediction(region, num_traces=100)  # Use all traces
        
        if region_stats and 'window_results' in region_stats:
            results.append(region_stats)
            
            # Print quick summary
            best_window = None
            best_correlation = -1
            
            for ws, stats in region_stats['window_results'].items():
                if stats['correlation'] > best_correlation:
                    best_correlation = stats['correlation']
                    best_window = ws
            
            if best_window:
                print(f" ✓ (best: {best_window}h window, r={best_correlation:.3f})")
            else:
                print(" ✓ (no correlation)")
        else:
            print(" ✗ (no data)")
    
    if not results:
        print("No valid data found. Exiting.")
        return
    
    # Create output directory
    output_dir = Path("outputs/sliding_window_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating analysis plots...")
    create_sliding_window_plot(results, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    
    # Print summary table
    print(f"\n📊 Summary (averaged across regions):")
    print(f"Window Size | Correlation | R-squared | MAE (hours) | Predictions")
    print(f"------------|-------------|-----------|-------------|------------")
    
    window_sizes = [1, 2, 4, 6, 12, 24, 48, 72]
    for ws in window_sizes:
        correlations = []
        r_squareds = []
        maes = []
        pred_counts = []
        
        for result in results:
            if ws in result['window_results']:
                correlations.append(result['window_results'][ws]['correlation'])
                r_squareds.append(result['window_results'][ws]['r_squared'])
                maes.append(result['window_results'][ws]['mae'])
                pred_counts.append(result['window_results'][ws]['total_predictions'])
        
        if correlations:
            avg_corr = np.mean(correlations)
            avg_r2 = np.mean(r_squareds)
            avg_mae = np.mean(maes)
            total_preds = sum(pred_counts)
            print(f"{ws:>10}h | {avg_corr:>10.3f} | {avg_r2:>8.3f} | {avg_mae:>10.3f} | {total_preds:>10d}")

if __name__ == "__main__":
    main()