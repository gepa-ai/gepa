#!/usr/bin/env python3
"""
Hybrid Prediction Model Development

Combines availability-based prediction with region historical data
to achieve better performance than either method alone.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class PredictionFeatures:
    """Features for hybrid prediction model"""
    availability_rate: float
    region_historical_mean: float
    region_historical_median: float
    recent_window_availability: float
    availability_trend: float  # change in availability over recent windows
    spot_variance_history: float  # historical variance in this region

class HybridPredictor:
    """Hybrid predictor combining multiple signals"""
    
    def __init__(self):
        self.availability_model = None
        self.region_stats = {}
        self.hybrid_model = None
        self.feature_importance = {}
        
    def extract_features(self, trace_data: List[int], region_id: str, 
                        observation_ratio: float = 0.3) -> Optional[PredictionFeatures]:
        """Extract all features for prediction"""
        
        if len(trace_data) < 50:  # Need minimum data
            return None
            
        # Split observation and prediction windows
        obs_end = int(len(trace_data) * observation_ratio)
        obs_window = trace_data[:obs_end]
        
        # Availability features
        availability_rate = 1 - np.mean(obs_window)
        
        # Recent window availability (last 20% of observation)
        recent_start = max(0, obs_end - int(obs_end * 0.2))
        recent_availability = 1 - np.mean(obs_window[recent_start:])
        
        # Availability trend
        if obs_end >= 20:
            early_avail = 1 - np.mean(obs_window[:obs_end//2])
            late_avail = 1 - np.mean(obs_window[obs_end//2:])
            trend = late_avail - early_avail
        else:
            trend = 0.0
            
        # Region historical stats
        region_mean = self.region_stats.get(region_id, {}).get('mean', 0)
        region_median = self.region_stats.get(region_id, {}).get('median', 0)
        region_variance = self.region_stats.get(region_id, {}).get('variance', 0)
        
        return PredictionFeatures(
            availability_rate=availability_rate,
            region_historical_mean=region_mean,
            region_historical_median=region_median,
            recent_window_availability=recent_availability,
            availability_trend=trend,
            spot_variance_history=region_variance
        )
    
    def compute_region_statistics(self, trace_files: List[str]):
        """Compute historical statistics for each region"""
        
        logger.info("Computing region historical statistics...")
        region_durations = {}
        
        for trace_file in trace_files:
            try:
                with open(trace_file, 'r') as f:
                    trace = json.load(f)
                
                region_id = Path(trace_file).parent.name
                durations = []
                
                # Calculate all spot durations in this trace
                data = trace['data']
                current_duration = 0
                
                for i, status in enumerate(data):
                    if status == 0:  # Available
                        current_duration += 1
                    else:  # Preempted
                        if current_duration > 0:
                            durations.append(current_duration)
                            current_duration = 0
                
                # Add final duration if trace ends available
                if current_duration > 0:
                    durations.append(current_duration)
                
                if durations:
                    if region_id not in region_durations:
                        region_durations[region_id] = []
                    region_durations[region_id].extend(durations)
                    
            except Exception as e:
                logger.warning(f"Error processing {trace_file}: {e}")
                continue
        
        # Compute statistics
        for region_id, durations in region_durations.items():
            if durations:
                self.region_stats[region_id] = {
                    'mean': np.mean(durations),
                    'median': np.median(durations),
                    'variance': np.var(durations),
                    'count': len(durations)
                }
        
        logger.info(f"Computed statistics for {len(self.region_stats)} regions")
    
    def fit_availability_model(self, features_list: List[PredictionFeatures], 
                             targets: List[float]):
        """Fit the availability-based component"""
        
        X_avail = np.array([[f.availability_rate, f.recent_window_availability, 
                           f.availability_trend] for f in features_list])
        y = np.array(targets)
        
        # Simple exponential model for availability
        def availability_predict(avail_rate):
            if avail_rate >= 0.7:
                return 8.5 * np.exp(2.1 * avail_rate)
            elif avail_rate >= 0.4:
                return 3.2 + 4.8 * avail_rate
            else:
                return 1.8 + 2.5 * avail_rate
        
        self.availability_model = availability_predict
        
    def fit_hybrid_model(self, features_list: List[PredictionFeatures], 
                        targets: List[float]):
        """Fit the hybrid model using weighted combination"""
        
        # Simple weighted combination approach
        X = []
        for f in features_list:
            # Availability prediction
            avail_pred = self.availability_model(f.availability_rate)
            
            features = [
                f.availability_rate,
                f.region_historical_mean,
                f.region_historical_median,
                f.recent_window_availability,
                f.availability_trend,
                f.spot_variance_history,
                avail_pred
            ]
            X.append(features)
        
        X = np.array(X)
        y = np.array(targets)
        
        # Find optimal weights using simple grid search
        best_weights = None
        best_mae = float('inf')
        
        # Test different weight combinations
        weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for w_avail in weight_options:
            for w_region in weight_options:
                for w_recent in weight_options:
                    if w_avail + w_region + w_recent == 0:
                        continue
                        
                    # Normalize weights
                    total_w = w_avail + w_region + w_recent
                    w_avail_norm = w_avail / total_w
                    w_region_norm = w_region / total_w
                    w_recent_norm = w_recent / total_w
                    
                    # Make predictions
                    predictions = []
                    for i, f in enumerate(features_list):
                        avail_pred = self.availability_model(f.availability_rate)
                        pred = (w_avail_norm * avail_pred + 
                               w_region_norm * f.region_historical_mean + 
                               w_recent_norm * f.recent_window_availability * 5.0)  # Scale recent availability
                        predictions.append(pred)
                    
                    # Calculate MAE
                    mae = np.mean(np.abs(np.array(predictions) - y))
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_weights = (w_avail_norm, w_region_norm, w_recent_norm)
        
        self.hybrid_weights = best_weights
        logger.info(f"Best weights: availability={best_weights[0]:.3f}, region={best_weights[1]:.3f}, recent={best_weights[2]:.3f}")
        logger.info(f"Training MAE: {best_mae:.3f}")
        
        # Store feature importance based on weights
        self.feature_importance = {
            'availability_prediction': best_weights[0],
            'region_historical_mean': best_weights[1], 
            'recent_availability': best_weights[2]
        }
    
    def predict(self, features: PredictionFeatures) -> float:
        """Make hybrid prediction"""
        
        if not hasattr(self, 'hybrid_weights'):
            return self.availability_model(features.availability_rate)
        
        # Get component predictions
        avail_pred = self.availability_model(features.availability_rate)
        region_pred = features.region_historical_mean
        recent_pred = features.recent_window_availability * 5.0  # Scale recent availability
        
        # Weighted combination
        w_avail, w_region, w_recent = self.hybrid_weights
        prediction = (w_avail * avail_pred + 
                     w_region * region_pred + 
                     w_recent * recent_pred)
        
        return max(0.1, prediction)  # Ensure positive prediction

def evaluate_hybrid_predictor():
    """Evaluate the hybrid prediction approach"""
    
    data_path = Path("/home/andyl/cant-be-late/data/converted_multi_region_aligned")
    trace_files = list(data_path.glob("*/0.json"))
    
    if not trace_files:
        logger.error(f"No trace files found in {data_path}")
        return
    
    logger.info(f"Found {len(trace_files)} trace files")
    
    # Initialize predictor
    predictor = HybridPredictor()
    
    # Compute region statistics
    predictor.compute_region_statistics(trace_files)
    
    # Collect training data
    features_list = []
    targets = []
    baseline_predictions = {
        'availability_only': [],
        'region_only': [],
        'overall_mean': [],
        'overall_median': []
    }
    
    all_durations = []
    
    for trace_file in trace_files:
        try:
            with open(trace_file, 'r') as f:
                trace = json.load(f)
            
            region_id = Path(trace_file).parent.name
            data = trace['data']
            
            # Calculate actual average duration in prediction window
            obs_end = int(len(data) * 0.3)
            pred_window = data[obs_end:]
            
            durations = []
            current_duration = 0
            
            for status in pred_window:
                if status == 0:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            if not durations:
                continue
                
            actual_avg = np.mean(durations)
            all_durations.extend(durations)
            
            # Extract features
            features = predictor.extract_features(data, region_id)
            if features is None:
                logger.warning(f"Failed to extract features for {trace_file}")
                continue
            
            features_list.append(features)
            targets.append(actual_avg)
            logger.debug(f"Added sample: {region_id}, target={actual_avg:.2f}, avail={features.availability_rate:.3f}")
            
            # Baseline predictions
            baseline_predictions['availability_only'].append(
                predictor.availability_model(features.availability_rate) if predictor.availability_model else 5.0
            )
            baseline_predictions['region_only'].append(features.region_historical_mean)
            baseline_predictions['overall_mean'].append(np.mean(all_durations) if all_durations else 5.0)
            baseline_predictions['overall_median'].append(np.median(all_durations) if all_durations else 5.0)
            
        except Exception as e:
            logger.warning(f"Error processing {trace_file}: {e}")
            continue
    
    if len(features_list) < 5:
        logger.error("Insufficient data for training")
        return
    
    logger.info(f"Collected {len(features_list)} training samples")
    
    # Fit models
    predictor.fit_availability_model(features_list, targets)
    predictor.fit_hybrid_model(features_list, targets)
    
    # Make hybrid predictions
    hybrid_predictions = [predictor.predict(f) for f in features_list]
    
    # Evaluate all methods
    targets = np.array(targets)
    methods = {
        'Hybrid Model': hybrid_predictions,
        'Availability Only': baseline_predictions['availability_only'],
        'Region History': baseline_predictions['region_only'],
        'Overall Mean': baseline_predictions['overall_mean'],
        'Overall Median': baseline_predictions['overall_median']
    }
    
    def calculate_r2(y_true, y_pred):
        """Calculate R² score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_mae(y_true, y_pred):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    results = {}
    
    print("\n" + "="*60)
    print("HYBRID PREDICTION MODEL EVALUATION")
    print("="*60)
    
    for method_name, predictions in methods.items():
        predictions = np.array(predictions)
        mae = calculate_mae(targets, predictions)
        r2 = calculate_r2(targets, predictions)
        
        results[method_name] = {'mae': mae, 'r2': r2}
        
        print(f"\n{method_name}:")
        print(f"  MAE: {mae:.3f} hours")
        print(f"  R²:  {r2:.4f}")
    
    # Calculate improvements
    hybrid_mae = results['Hybrid Model']['mae']
    availability_mae = results['Availability Only']['mae']
    region_mae = results['Region History']['mae']
    
    vs_availability = (availability_mae - hybrid_mae) / availability_mae * 100
    vs_region = (region_mae - hybrid_mae) / region_mae * 100
    
    print(f"\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    print(f"Hybrid vs Availability Only: {vs_availability:+.1f}%")
    print(f"Hybrid vs Region History:    {vs_region:+.1f}%")
    
    # Feature importance
    if predictor.feature_importance:
        print(f"\n" + "="*60)
        print("FEATURE IMPORTANCE")
        print("="*60)
        for name, importance in sorted(predictor.feature_importance.items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"{name:25}: {importance:.4f}")
    
    # Visualization
    create_hybrid_evaluation_plots(targets, methods, results)
    
    return results

def create_hybrid_evaluation_plots(targets, methods, results):
    """Create comprehensive evaluation plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction vs Actual scatter
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (method, predictions) in enumerate(methods.items()):
        if method == 'Hybrid Model':
            ax1.scatter(targets, predictions, alpha=0.7, color=colors[i], 
                       label=f'{method} (R²={results[method]["r2"]:.3f})', s=50)
        else:
            ax1.scatter(targets, predictions, alpha=0.4, color=colors[i], 
                       label=f'{method} (R²={results[method]["r2"]:.3f})', s=20)
    
    max_val = max(np.max(targets), max([np.max(p) for p in methods.values()]))
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax1.set_xlabel('Actual Duration (hours)')
    ax1.set_ylabel('Predicted Duration (hours)')
    ax1.set_title('Prediction Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE comparison bar chart
    ax2 = axes[0, 1]
    method_names = list(results.keys())
    maes = [results[m]['mae'] for m in method_names]
    bars = ax2.bar(range(len(method_names)), maes, color=colors[:len(method_names)])
    
    # Highlight hybrid model
    bars[0].set_color('red')
    bars[0].set_alpha(0.8)
    
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Mean Absolute Error (hours)')
    ax2.set_title('Prediction Error Comparison')
    ax2.set_xticks(range(len(method_names)))
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add MAE values on bars
    for i, mae in enumerate(maes):
        ax2.text(i, mae + 0.1, f'{mae:.2f}', ha='center', va='bottom')
    
    # 3. Residuals plot for hybrid model
    ax3 = axes[1, 0]
    hybrid_pred = np.array(methods['Hybrid Model'])
    residuals = targets - hybrid_pred
    
    ax3.scatter(hybrid_pred, residuals, alpha=0.6)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Predicted Duration (hours)')
    ax3.set_ylabel('Residuals (Actual - Predicted)')
    ax3.set_title('Hybrid Model Residuals')
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement percentage
    ax4 = axes[1, 1]
    hybrid_mae = results['Hybrid Model']['mae']
    improvements = []
    method_labels = []
    
    for method, result in results.items():
        if method != 'Hybrid Model':
            improvement = (result['mae'] - hybrid_mae) / result['mae'] * 100
            improvements.append(improvement)
            method_labels.append(method)
    
    bars = ax4.bar(range(len(improvements)), improvements, 
                   color=['green' if x > 0 else 'red' for x in improvements])
    ax4.set_xlabel('Baseline Method')
    ax4.set_ylabel('Improvement over Baseline (%)')
    ax4.set_title('Hybrid Model Improvement')
    ax4.set_xticks(range(len(method_labels)))
    ax4.set_xticklabels(method_labels, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    
    # Add improvement values
    for i, imp in enumerate(improvements):
        ax4.text(i, imp + (1 if imp > 0 else -1), f'{imp:+.1f}%', 
                ha='center', va='bottom' if imp > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('/home/andyl/cant-be-late/scripts_multi/trace_sampling/spot_duration_prediction_analysis/outputs/hybrid_model_evaluation.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Evaluation plots saved to outputs/hybrid_model_evaluation.png")

if __name__ == "__main__":
    evaluate_hybrid_predictor()