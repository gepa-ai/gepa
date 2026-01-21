#!/usr/bin/env python3
"""
Calculate actual prediction improvement from availability-based prediction.
Compare our method vs simple baselines to quantify the practical value.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from validate_trace_level_prediction import validate_trace_level_prediction

def calculate_baseline_methods(predictions):
    """Calculate various baseline prediction methods."""
    actual_durations = [p['subsequent_avg_duration'] for p in predictions]
    
    baselines = {}
    
    # 1. Always predict the overall mean
    overall_mean = np.mean(actual_durations)
    mean_predictions = [overall_mean] * len(actual_durations)
    baselines['overall_mean'] = {
        'predictions': mean_predictions,
        'mae': np.mean(np.abs(np.array(actual_durations) - np.array(mean_predictions))),
        'description': f'Always predict {overall_mean:.1f}h (overall mean)'
    }
    
    # 2. Always predict the median
    overall_median = np.median(actual_durations)
    median_predictions = [overall_median] * len(actual_durations)
    baselines['overall_median'] = {
        'predictions': median_predictions,
        'mae': np.mean(np.abs(np.array(actual_durations) - np.array(median_predictions))),
        'description': f'Always predict {overall_median:.1f}h (overall median)'
    }
    
    # 3. Random prediction within observed range
    min_dur, max_dur = min(actual_durations), max(actual_durations)
    np.random.seed(42)  # For reproducibility
    random_predictions = np.random.uniform(min_dur, max_dur, len(actual_durations))
    baselines['random'] = {
        'predictions': random_predictions.tolist(),
        'mae': np.mean(np.abs(np.array(actual_durations) - random_predictions)),
        'description': f'Random between {min_dur:.1f}-{max_dur:.1f}h'
    }
    
    # 4. Region-based mean (predict based on region historical average)
    region_means = {}
    for p in predictions:
        region = p['region']
        if region not in region_means:
            region_means[region] = []
        region_means[region].append(p['subsequent_avg_duration'])
    
    # Calculate region means
    for region in region_means:
        region_means[region] = np.mean(region_means[region])
    
    region_predictions = [region_means[p['region']] for p in predictions]
    baselines['region_mean'] = {
        'predictions': region_predictions,
        'mae': np.mean(np.abs(np.array(actual_durations) - np.array(region_predictions))),
        'description': 'Predict based on region historical mean'
    }
    
    return baselines

def calculate_availability_based_prediction(predictions):
    """Calculate our availability-based prediction method."""
    obs_availabilities = [p['observation_availability'] for p in predictions]
    actual_durations = [p['subsequent_avg_duration'] for p in predictions]
    
    # Fit a simple model (same as we found before)
    # Use piecewise constant model based on availability segments
    segments = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.0)]
    segment_means = {}
    
    for min_avail, max_avail in segments:
        segment_data = [p['subsequent_avg_duration'] for p in predictions 
                       if min_avail <= p['observation_availability'] < max_avail]
        if segment_data:
            segment_means[(min_avail, max_avail)] = np.mean(segment_data)
        else:
            segment_means[(min_avail, max_avail)] = np.mean(actual_durations)  # fallback
    
    # Make predictions using our method
    our_predictions = []
    for p in predictions:
        avail = p['observation_availability']
        prediction = None
        for (min_avail, max_avail), mean_dur in segment_means.items():
            if min_avail <= avail < max_avail:
                prediction = mean_dur
                break
        if prediction is None:  # Handle edge case
            prediction = np.mean(actual_durations)
        our_predictions.append(prediction)
    
    our_mae = np.mean(np.abs(np.array(actual_durations) - np.array(our_predictions)))
    
    return {
        'predictions': our_predictions,
        'mae': our_mae,
        'segment_means': segment_means,
        'description': 'Availability-based segmented prediction'
    }

def calculate_improvements(our_method, baselines, actual_durations):
    """Calculate improvement metrics."""
    improvements = {}
    
    our_mae = our_method['mae']
    
    for name, baseline in baselines.items():
        baseline_mae = baseline['mae']
        
        # Calculate improvement
        abs_improvement = baseline_mae - our_mae
        rel_improvement = abs_improvement / baseline_mae * 100
        
        improvements[name] = {
            'baseline_mae': baseline_mae,
            'our_mae': our_mae,
            'absolute_improvement': abs_improvement,
            'relative_improvement': rel_improvement,
            'description': baseline['description']
        }
    
    return improvements

def create_improvement_visualization(predictions, our_method, baselines, improvements, output_dir):
    """Create comprehensive improvement visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    actual_durations = [p['subsequent_avg_duration'] for p in predictions]
    obs_availabilities = [p['observation_availability'] for p in predictions]
    
    # Plot 1: Our method vs actual
    ax1 = axes[0, 0]
    ax1.scatter(actual_durations, our_method['predictions'], alpha=0.6, s=30, color='blue')
    
    # Perfect prediction line
    min_val = min(min(actual_durations), min(our_method['predictions']))
    max_val = max(max(actual_durations), max(our_method['predictions']))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
    
    # Calculate R²
    correlation, _ = pearsonr(actual_durations, our_method['predictions'])
    r2 = correlation ** 2
    
    ax1.set_xlabel('Actual Duration (hours)')
    ax1.set_ylabel('Predicted Duration (hours)')
    ax1.set_title(f'Our Method: Availability-Based Prediction\nMAE={our_method["mae"]:.2f}h, r={correlation:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAE comparison
    ax2 = axes[0, 1]
    
    methods = ['Our Method'] + list(baselines.keys())
    maes = [our_method['mae']] + [baselines[name]['mae'] for name in baselines.keys()]
    colors = ['blue'] + ['gray'] * len(baselines)
    
    bars = ax2.bar(range(len(methods)), maes, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mae:.2f}h', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Prediction Method')
    ax2.set_ylabel('Mean Absolute Error (hours)')
    ax2.set_title('Prediction Accuracy Comparison')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Improvement percentages
    ax3 = axes[1, 0]
    
    baseline_names = list(improvements.keys())
    rel_improvements = [improvements[name]['relative_improvement'] for name in baseline_names]
    colors_imp = ['green' if imp > 0 else 'red' for imp in rel_improvements]
    
    bars = ax3.bar(range(len(baseline_names)), rel_improvements, color=colors_imp, alpha=0.7)
    
    # Add value labels
    for bar, imp in zip(bars, rel_improvements):
        height = bar.get_height()
        y_pos = height + 0.5 if height >= 0 else height - 0.5
        ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Baseline Method')
    ax3.set_ylabel('Relative Improvement (%)')
    ax3.set_title('Prediction Improvement vs Baselines')
    ax3.set_xticks(range(len(baseline_names)))
    ax3.set_xticklabels(baseline_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Availability-based segmentation
    ax4 = axes[1, 1]
    
    # Show our segmentation approach
    segments = [(0.0, 0.4, 'Low'), (0.4, 0.7, 'Med'), (0.7, 1.0, 'High')]
    segment_colors = ['red', 'orange', 'green']
    
    for i, (min_avail, max_avail, label) in enumerate(segments):
        segment_preds = [p for p in predictions 
                        if min_avail <= p['observation_availability'] < max_avail]
        
        if segment_preds:
            x_vals = [p['observation_availability'] for p in segment_preds]
            y_vals = [p['subsequent_avg_duration'] for p in segment_preds]
            
            ax4.scatter(x_vals, y_vals, alpha=0.6, s=20, color=segment_colors[i], 
                       label=f'{label} (n={len(segment_preds)})')
            
            # Show segment prediction (horizontal line)
            segment_mean = our_method['segment_means'][(min_avail, max_avail)]
            ax4.hlines(segment_mean, min_avail, max_avail, colors=segment_colors[i], 
                      linestyle='-', linewidth=3, alpha=0.8)
    
    ax4.set_xlabel('Observation Availability')
    ax4.set_ylabel('Actual Subsequent Duration (hours)')
    ax4.set_title('Our Segmentation Strategy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_improvement_summary(improvements, our_method):
    """Print comprehensive improvement summary."""
    print(f"\n" + "="*80)
    print("PREDICTION IMPROVEMENT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Our Method Performance:")
    print(f"   MAE: {our_method['mae']:.2f} hours")
    print(f"   Method: {our_method['description']}")
    
    print(f"\n📈 Improvement vs Baselines:")
    print("-"*80)
    print(f"{'Baseline Method':<25} {'Baseline MAE':<12} {'Our MAE':<10} {'Abs Improv':<12} {'Rel Improv':<12}")
    print("-"*80)
    
    for name, imp in improvements.items():
        print(f"{name:<25} {imp['baseline_mae']:<12.2f} {imp['our_mae']:<10.2f} "
              f"{imp['absolute_improvement']:<12.2f} {imp['relative_improvement']:<+12.1f}%")
    
    print(f"\n🎯 Key Insights:")
    
    # Find best improvement
    best_improvement = max(improvements.values(), key=lambda x: x['relative_improvement'])
    best_baseline = [k for k, v in improvements.items() if v == best_improvement][0]
    
    print(f"   • Best improvement: {best_improvement['relative_improvement']:+.1f}% vs {best_baseline}")
    
    # Overall assessment
    avg_improvement = np.mean([imp['relative_improvement'] for imp in improvements.values()])
    
    if avg_improvement > 20:
        assessment = "EXCELLENT"
    elif avg_improvement > 10:
        assessment = "GOOD"
    elif avg_improvement > 5:
        assessment = "MODERATE"
    else:
        assessment = "LIMITED"
    
    print(f"   • Average improvement: {avg_improvement:+.1f}% ({assessment})")
    
    # Practical value
    best_abs_improvement = max([imp['absolute_improvement'] for imp in improvements.values()])
    print(f"   • Best absolute improvement: {best_abs_improvement:.2f} hours")
    
    return avg_improvement

def main():
    """Calculate and analyze prediction improvements."""
    print("="*80)
    print("CALCULATING PREDICTION IMPROVEMENT")
    print("="*80)
    
    # Get predictions data
    print("Loading trace-level predictions...")
    predictions = validate_trace_level_prediction()
    
    if not predictions:
        print("No predictions available!")
        return
    
    print(f"Analyzing {len(predictions)} predictions...")
    
    # Calculate baselines
    print("\nCalculating baseline methods...")
    baselines = calculate_baseline_methods(predictions)
    
    # Calculate our method
    print("Calculating our availability-based method...")
    our_method = calculate_availability_based_prediction(predictions)
    
    # Calculate improvements
    actual_durations = [p['subsequent_avg_duration'] for p in predictions]
    improvements = calculate_improvements(our_method, baselines, actual_durations)
    
    # Create output directory
    output_dir = Path("outputs/prediction_improvement")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualization
    print("Generating improvement analysis...")
    create_improvement_visualization(predictions, our_method, baselines, improvements, output_dir)
    
    # Print summary
    avg_improvement = print_improvement_summary(improvements, our_method)
    
    # Save results
    results = {
        'our_method': {k: v for k, v in our_method.items() if k != 'predictions'},
        'improvements': improvements,
        'average_improvement': avg_improvement
    }
    
    with open(output_dir / 'improvement_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()