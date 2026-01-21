#!/usr/bin/env python3
"""
Batch comparison script for AAEV vs baseline strategies across multiple traces.
Generates both results and visualizations for comprehensive analysis.
"""

import subprocess
import json
import os
from pathlib import Path
import time

# Default test parameters
DEFAULT_TASK_DURATION_HOURS = 30.0
DEFAULT_DEADLINE_HOURS = 52.0
DEFAULT_RESTART_OVERHEAD_HOURS = 0.2

# Default regions to test
DEFAULT_REGIONS = [
    "us-east-1a_v100_1",
    "us-east-1d_v100_1",
    "us-east-1c_v100_1",
    "us-east-1f_v100_1",
    "us-east-2a_v100_1"
]

def run_strategy_test(strategy_name, trace_ids, output_dir, timeout=300, strategy_files=None,
                     task_duration_hours=DEFAULT_TASK_DURATION_HOURS,
                     deadline_hours=DEFAULT_DEADLINE_HOURS,
                     restart_overhead_hours=DEFAULT_RESTART_OVERHEAD_HOURS):
    """Run strategy test for multiple trace combinations."""
    results = []
    
    for trace_id in trace_ids:
        print(f"\n=== Testing {strategy_name} on trace {trace_id} ===")
        
        # Build trace file paths
        trace_files = [
            f"data/converted_multi_region_aligned/{region}/{trace_id}.json"
            for region in DEFAULT_REGIONS
        ]
        
        # Check if all trace files exist
        if not all(Path(f).exists() for f in trace_files):
            print(f"Skipping trace {trace_id} - missing files")
            continue
            
        # Run main.py directly
        test_output_dir = f"{output_dir}/{strategy_name}_trace_{trace_id}"
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build command - support both strategy names and strategy files
        cmd = ["python", "main.py"]
        
        if strategy_files and strategy_name in strategy_files:
            # Use strategy file
            cmd.append(f"--strategy-file={strategy_files[strategy_name]}")
        else:
            # Use strategy name (registered strategy)
            cmd.append(f"--strategy={strategy_name}")
        
        cmd.extend([
            "--env=multi_trace",
            "--trace-files"] + trace_files + [
            f"--task-duration-hours={task_duration_hours}",
            f"--deadline-hours={deadline_hours}",
            f"--restart-overhead-hours={restart_overhead_hours}",
            f"--output-dir={test_output_dir}"
        ])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                print(f"✅ {strategy_name} trace {trace_id} completed")
                
                # Build the expected filename based on the parameters we used
                # This matches the format from simulate.py
                test_output_path = Path(test_output_dir)
                trace_names = ','.join([f"{trace_id}.json"] * len(DEFAULT_REGIONS))
                expected_filename = (f"{strategy_name}-multi_trace-multi_region_{trace_names}-"
                                   f"ddl={deadline_hours}-task=SingleTask({task_duration_hours}h)-"
                                   f"over={restart_overhead_hours}")
                expected_file = test_output_path / expected_filename
                
                # No fallback - fail immediately if file doesn't exist
                with open(expected_file, 'r') as f:
                    data = json.load(f)
                    # Cost is in 'costs' array, take the first (and only) value
                    cost = data.get('costs', [None])[0] if data.get('costs') else data.get('Cost', 'N/A')
                    
                results.append({
                    'trace_id': trace_id,
                    'strategy': strategy_name,
                    'cost': cost,
                    'status': 'success',
                    'output_dir': test_output_dir,
                    'result_file': str(expected_file)
                })
            else:
                print(f"❌ {strategy_name} trace {trace_id} failed: {result.stderr}")
                results.append({
                    'trace_id': trace_id,
                    'strategy': strategy_name,
                    'cost': 'N/A',
                    'status': 'failed',
                    'error': result.stderr[:200]
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {strategy_name} trace {trace_id} timed out")
            results.append({
                'trace_id': trace_id,
                'strategy': strategy_name,
                'cost': 'N/A',
                'status': 'timeout'
            })
        except Exception as e:
            print(f"💥 {strategy_name} trace {trace_id} error: {e}")
            results.append({
                'trace_id': trace_id,
                'strategy': strategy_name,
                'cost': 'N/A',
                'status': 'error',
                'error': str(e)
            })
    
    return results

def generate_visualizations(results, output_dir):
    """Generate visualizations for successful results."""
    viz_results = []
    generated_files = []
    
    print(f"\n{'='*50}")
    print("Generating Timeline Visualizations...")
    print(f"{'='*50}")
    
    for result in results:
        if result['status'] == 'success' and 'result_file' in result:
            viz_file = f"{output_dir}/viz_{result['strategy']}_trace_{result['trace_id']}.png"
            
            cmd = [
                "python", "scripts_multi/visualize_timeline_segments.py",
                result['result_file'],
                "-o", viz_file
            ]
            
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = "/home/andyl/cant-be-late"
                
                viz_result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
                
                if viz_result.returncode == 0:
                    generated_files.append((result['strategy'], result['trace_id'], viz_file))
                    viz_results.append({
                        **result,
                        'viz_file': viz_file,
                        'viz_status': 'success'
                    })
                else:
                    print(f"❌ Failed: {result['strategy']} trace {result['trace_id']}")
                    viz_results.append({
                        **result,
                        'viz_status': 'failed',
                        'viz_error': viz_result.stderr[:200]
                    })
                    
            except Exception as e:
                print(f"💥 Error: {result['strategy']} trace {result['trace_id']}: {e}")
                viz_results.append({
                    **result,
                    'viz_status': 'error',
                    'viz_error': str(e)
                })
        else:
            viz_results.append(result)
    
    # Print what was generated
    if generated_files:
        print(f"\n✅ Generated {len(generated_files)} visualizations:")
        for strategy, trace_id, file_path in generated_files:
            print(f"   • {strategy} trace {trace_id}: {file_path}")
    else:
        print("\n⚠️ No visualizations generated")
    
    return viz_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch strategy comparison tool')
    parser.add_argument('num_traces', type=int, nargs='?', default=5, 
                       help='Number of traces to test (default: 5)')
    parser.add_argument('--strategies', nargs='+', 
                       default=["availability_aware_expected_value", "multi_region_rc_cr_threshold"],
                       help='Strategies to compare (default: AAEV vs baseline)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: outputs/multi_trace_comparison_Ntraces)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per test in seconds (default: 300)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip generating visualizations to save time')
    parser.add_argument('--strategy-files', nargs='*', metavar='STRATEGY=FILE',
                       help='Map strategy names to files (format: strategy_name=file_path)')
    
    args = parser.parse_args()
    
    # Parse strategy files mapping
    if args.strategy_files:
        strategy_files = {}
        for mapping in args.strategy_files:
            if '=' in mapping:
                strategy_name, file_path = mapping.split('=', 1)
                strategy_files[strategy_name] = file_path
            else:
                print(f"Warning: Invalid strategy file mapping format: {mapping}")
        args.strategy_files = strategy_files
    else:
        args.strategy_files = None
    
    # Test parameters
    num_traces = args.num_traces
    trace_ids = list(range(num_traces))
    strategies = args.strategies
    output_base_dir = args.output_dir or f"outputs/multi_trace_comparison_{num_traces}traces"
    
    # Create output directory
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting batch strategy comparison...")
    print(f"Trace IDs: {trace_ids}")
    print(f"Strategies: {strategies}")
    
    all_results = []
    
    # Run each strategy
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Running strategy: {strategy}")
        print(f"{'='*50}")
        
        strategy_results = run_strategy_test(strategy, trace_ids, output_base_dir, args.timeout, getattr(args, 'strategy_files', None))
        all_results.extend(strategy_results)
    
    # Print cost summary FIRST
    print(f"\n{'='*50}")
    print("COST SUMMARY")
    print(f"{'='*50}")
    
    for strategy in strategies:
        strategy_results = [r for r in all_results if r['strategy'] == strategy]
        successful = [r for r in strategy_results if r['status'] == 'success']
        costs = [r['cost'] for r in successful if isinstance(r['cost'], (int, float))]
        
        print(f"\n{strategy}:")
        print(f"  Successful runs: {len(successful)}/{len(strategy_results)}")
        if costs:
            print(f"  Average cost: ${sum(costs)/len(costs):.2f}")
            print(f"  Cost range: ${min(costs):.2f} - ${max(costs):.2f}")
            print(f"  Individual costs:")
            for r in successful:
                if isinstance(r['cost'], (int, float)):
                    print(f"    Trace {r['trace_id']}: ${r['cost']:.2f}")
    
    # Generate visualizations (unless disabled)
    if not args.no_viz:
        print(f"\n{'='*50}")
        print("Generating visualizations...")
        print(f"{'='*50}")
        
        final_results = generate_visualizations(all_results, output_base_dir)
    else:
        print("\nSkipping visualizations (--no-viz)")
        final_results = all_results
    
    # Save summary report
    summary_file = f"{output_base_dir}/comparison_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print final summary (shorter version)
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    
    for strategy in strategies:
        strategy_results = [r for r in final_results if r['strategy'] == strategy]
        successful = [r for r in strategy_results if r['status'] == 'success']
        costs = [r['cost'] for r in successful if isinstance(r['cost'], (int, float))]
        
        print(f"\n{strategy}:")
        print(f"  Successful runs: {len(successful)}/{len(strategy_results)}")
        if costs:
            print(f"  Average cost: ${sum(costs)/len(costs):.2f}")
            print(f"  Cost range: ${min(costs):.2f} - ${max(costs):.2f}")
    
    print(f"\n📊 Output files:")
    print(f"   • Results: {summary_file}")
    
    # List visualization files
    viz_files = [r.get('viz_file') for r in final_results if 'viz_file' in r]
    if viz_files:
        print(f"   • Visualizations: {len(viz_files)} PNG files in {output_base_dir}/")
        print(f"     (viz_<strategy>_trace_<id>.png)")
    
    print(f"\n💡 To view a visualization: open {output_base_dir}/viz_<strategy>_trace_<id>.png")

if __name__ == "__main__":
    main()