#!/usr/bin/env python3
"""
Quick strategy testing script for Phase 1 development.
Tests a single strategy on one trace and immediately visualizes results.
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def main():
    # Parse arguments
    args = sys.argv[1:]
    verbose = False
    
    # Check for --verbose flag
    if '--verbose' in args:
        verbose = True
        args.remove('--verbose')
    
    if len(args) < 1:
        print("Usage: python quick_test_strategy.py <strategy_name> [trace_id] [--verbose]")
        print("Example: python quick_test_strategy.py availability_aware_expected_value")
        print("Example: python quick_test_strategy.py my_new_strategy 1")
        print("Example: python quick_test_strategy.py my_new_strategy --verbose")
        print("\nOptions:")
        print("  --verbose  Show real-time output from main.py")
        sys.exit(1)
    
    strategy_name = args[0]
    trace_id = args[1] if len(args) > 1 else "0"
    
    print(f"🚀 Quick testing strategy: {strategy_name}")
    print(f"📊 Using trace ID: {trace_id}")
    
    # Standard 5-region trace files
    trace_files = [
        f"data/converted_multi_region_aligned/us-east-1a_v100_1/{trace_id}.json",
        f"data/converted_multi_region_aligned/us-east-1d_v100_1/{trace_id}.json", 
        f"data/converted_multi_region_aligned/us-east-1c_v100_1/{trace_id}.json",
        f"data/converted_multi_region_aligned/us-east-1f_v100_1/{trace_id}.json",
        f"data/converted_multi_region_aligned/us-east-2a_v100_1/{trace_id}.json"
    ]
    
    # Check if trace files exist
    missing_files = [f for f in trace_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing trace files: {missing_files}")
        sys.exit(1)
    
    # Output directory
    output_dir = f"outputs/quick_test_{strategy_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Run the strategy
    cmd = [
        "python", "main.py",
        f"--strategy={strategy_name}",
        "--env=multi_trace",
        "--trace-files"] + trace_files + [
        "--task-duration-hours=30",
        "--deadline-hours=52", 
        "--restart-overhead-hours=0.2",
        f"--output-dir={output_dir}"
    ]
    
    print("⏳ Running strategy test...")
    if verbose:
        print("📺 Verbose mode: showing real-time output\n" + "="*50)
    
    try:
        if verbose:
            # Run with real-time output
            result = subprocess.run(cmd, timeout=120)
            returncode = result.returncode
        else:
            # Run with captured output (original behavior)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            returncode = result.returncode
            
        if returncode != 0:
            print(f"❌ Strategy test failed!")
            if not verbose:
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
            sys.exit(1)
        
        if verbose:
            print("="*50 + "\n")
        print("✅ Strategy test completed successfully!")
        
    except subprocess.TimeoutExpired:
        print("⏰ Strategy test timed out (>120s)")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Error running strategy: {e}")
        sys.exit(1)
    
    # Find result file
    result_files = [f for f in Path(output_dir).iterdir() 
                   if f.name.startswith(strategy_name) and not f.name.endswith('.log')]
    
    if not result_files:
        print("❌ No result file found!")
        sys.exit(1)
    
    result_file = result_files[0]
    
    # Extract cost
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            cost = data.get('costs', [None])[0] if data.get('costs') else data.get('Cost', 'N/A')
        
        print(f"💰 Final cost: ${cost}")
        
    except Exception as e:
        print(f"⚠️ Could not extract cost: {e}")
        cost = 'N/A'
    
    # Generate visualization
    viz_file = f"quick_test_{strategy_name}_trace{trace_id}.png"
    viz_cmd = [
        "python", "scripts_multi/visualize_timeline_segments.py",
        str(result_file),
        "-o", viz_file
    ]
    
    print("🎨 Generating visualization...")
    if verbose:
        print("="*50)
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/andyl/cant-be-late"
        
        if verbose:
            # Show visualization output in verbose mode
            viz_result = subprocess.run(viz_cmd, timeout=60, env=env)
            print("="*50)
            if viz_result.returncode == 0:
                print(f"✅ Visualization saved: {viz_file}")
            else:
                print(f"⚠️ Visualization failed")
        else:
            # Original behavior - capture output
            viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=60, env=env)
            if viz_result.returncode == 0:
                # Print the structured summary from visualization
                if viz_result.stdout:
                    print(viz_result.stdout)
                print(f"✅ Visualization saved: {viz_file}")
            else:
                print(f"⚠️ Visualization failed: {viz_result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("QUICK TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Strategy: {strategy_name}")
    print(f"Trace ID: {trace_id}")
    print(f"Cost: ${cost}")
    print(f"Result file: {result_file}")
    print(f"Visualization: {viz_file}")
    print(f"\n🔍 Next steps:")
    print(f"  1. Check {viz_file} for strategy behavior")
    print(f"  2. If good, run: python batch_strategy_comparison.py 3")
    print(f"  3. If excellent, run: python batch_strategy_comparison.py 100")

if __name__ == "__main__":
    main()