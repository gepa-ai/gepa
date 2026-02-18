#!/usr/bin/env python3
"""
Convert various data formats to multi-region simulation format.
Supports converting CSV availability data to JSON format expected by simulator.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConverter:
    def __init__(self):
        self.base_path = Path("data/real")
        
    def convert_csv_to_simulator_format(self, csv_file: str, output_dir: str, 
                                       num_traces: int = 100, trace_length_hours: int = 60):
        """
        Convert CSV availability data to simulator JSON format.
        
        Args:
            csv_file: Path to CSV file with columns [date, availability]
            output_dir: Output directory for JSON files
            num_traces: Number of different trace files to generate
            trace_length_hours: Length of each trace in hours
        """
        logger.info(f"Converting {csv_file} to simulator format...")
        
        # Read CSV data
        df = pd.read_csv(csv_file)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate gap_seconds from first two timestamps
        if len(df) >= 2:
            gap_seconds = int((df['date'].iloc[1] - df['date'].iloc[0]).total_seconds())
        else:
            gap_seconds = 600  # Default 10 minutes
            
        logger.info(f"Detected gap_seconds: {gap_seconds}")
        
        # Calculate trace length in ticks
        trace_length_ticks = int(trace_length_hours * 3600 / gap_seconds)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate multiple traces with different starting points
        max_start_index = max(0, len(df) - trace_length_ticks)
        
        if max_start_index <= 0:
            logger.warning(f"CSV data too short for {trace_length_hours}h traces. Using all data.")
            trace_length_ticks = len(df)
            max_start_index = 1
        
        for i in range(num_traces):
            # Random start index
            if max_start_index > 0:
                start_idx = np.random.randint(0, max_start_index)
            else:
                start_idx = 0
            
            end_idx = min(start_idx + trace_length_ticks, len(df))
            
            # Extract trace data
            trace_data = df['availability'].iloc[start_idx:end_idx].tolist()
            
            # Create JSON structure
            json_data = {
                "metadata": {
                    "gap_seconds": gap_seconds,
                    "start_time": df['date'].iloc[start_idx].isoformat(),
                    "source_file": csv_file
                },
                "data": trace_data
            }
            
            # Save JSON file
            output_file = os.path.join(output_dir, f"{i}.json")
            with open(output_file, 'w') as f:
                json.dump(json_data, f)
        
        logger.info(f"Generated {num_traces} trace files in {output_dir}")
        logger.info(f"Each trace: {len(trace_data)} ticks = {len(trace_data) * gap_seconds / 3600:.1f} hours")

    def convert_all_csv_data(self, csv_base_dir: str = "data/real/availability/2023-02-15/processed",
                            output_base_dir: str = "data/converted_multi_region"):
        """Convert all CSV files to simulator format."""
        csv_path = Path(csv_base_dir)
        if not csv_path.exists():
            logger.error(f"CSV directory not found: {csv_path}")
            return
        
        converted_regions = []
        
        for csv_file in csv_path.glob("*.csv"):
            # Extract region info from filename (e.g., us-east-1a_v100_1.csv)
            region_name = csv_file.stem
            output_dir = os.path.join(output_base_dir, region_name)
            
            try:
                self.convert_csv_to_simulator_format(str(csv_file), output_dir)
                converted_regions.append((region_name, output_dir))
            except Exception as e:
                logger.error(f"Failed to convert {csv_file}: {e}")
        
        logger.info(f"\nConversion complete! Converted {len(converted_regions)} regions:")
        for region_name, output_dir in converted_regions:
            logger.info(f"  {region_name} -> {output_dir}")
        
        return converted_regions

    def analyze_converted_data(self, converted_regions: List[Tuple[str, str]]):
        """Analyze availability of converted regions."""
        logger.info(f"\n{'='*60}")
        logger.info("AVAILABILITY ANALYSIS")
        logger.info(f"{'='*60}")
        
        analysis_results = []
        
        for region_name, region_dir in converted_regions:
            try:
                # Read first trace file for analysis
                with open(os.path.join(region_dir, "0.json")) as f:
                    data = json.load(f)
                
                trace_data = data['data']
                availability = 1 - (sum(trace_data) / len(trace_data))
                
                analysis_results.append({
                    'region': region_name,
                    'availability': availability,
                    'preemption_rate': 1 - availability,
                    'path': region_dir
                })
                
                logger.info(f"{region_name:25s}: {availability:.3f} availability, {1-availability:.3f} preemption")
                
            except Exception as e:
                logger.warning(f"Could not analyze {region_name}: {e}")
        
        return analysis_results

    def suggest_region_pairs(self, analysis_results: List[Dict]):
        """Suggest good region pairs for multi-region testing."""
        logger.info(f"\n{'='*60}")
        logger.info("SUGGESTED REGION PAIRS")
        logger.info(f"{'='*60}")
        
        # Group by instance type
        instance_types = {}
        for result in analysis_results:
            # Extract instance type from region name (e.g., us-east-1a_v100_1 -> v100_1)
            parts = result['region'].split('_')
            if len(parts) >= 2:
                instance_type = '_'.join(parts[-2:])  # e.g., v100_1
                if instance_type not in instance_types:
                    instance_types[instance_type] = []
                instance_types[instance_type].append(result)
        
        suggested_pairs = []
        
        for instance_type, regions in instance_types.items():
            if len(regions) < 2:
                continue
                
            logger.info(f"\nInstance type: {instance_type}")
            
            # Sort by availability
            regions.sort(key=lambda x: x['availability'])
            
            # Suggest pairs with different availability levels
            for i in range(len(regions)):
                for j in range(i+1, len(regions)):
                    region1, region2 = regions[i], regions[j]
                    avail_diff = abs(region1['availability'] - region2['availability'])
                    
                    # Good pairs have significant availability difference
                    if avail_diff > 0.1:  # At least 10% difference
                        suggested_pairs.append({
                            'instance_type': instance_type,
                            'region1': region1['region'],
                            'region1_avail': region1['availability'],
                            'region1_path': region1['path'],
                            'region2': region2['region'], 
                            'region2_avail': region2['availability'],
                            'region2_path': region2['path'],
                            'avail_diff': avail_diff
                        })
                        
                        logger.info(f"  {region1['region']:20s} ({region1['availability']:.3f}) + "
                                   f"{region2['region']:20s} ({region2['availability']:.3f}) "
                                   f"[diff: {avail_diff:.3f}]")
        
        return suggested_pairs

    def generate_multi_region_test_script(self, suggested_pairs: List[Dict]):
        """Generate test script for suggested region pairs."""
        script_content = """#!/bin/bash
# Auto-generated multi-region test script

export LOG_LEVEL=INFO
export RESTART_OVERHEAD_HOURS=0.2

echo "Starting multi-region evaluation for converted data..."

"""
        
        for i, pair in enumerate(suggested_pairs):
            instance_type = pair['instance_type']
            region1_path = pair['region1_path']
            region2_path = pair['region2_path']
            
            script_content += f"""
echo "Testing {instance_type}: {pair['region1']} + {pair['region2']}"

# Single region tests
python ./main.py --strategy=rc_cr_threshold \\
    --env trace \\
    --trace-file {region1_path} \\
    --restart-overhead-hours=0.2 \\
    --deadline-hours=52 \\
    --output-dir exp-converted-{instance_type}-region1 \\
    --task-duration-hours=48

python ./main.py --strategy=rc_cr_threshold \\
    --env trace \\
    --trace-file {region2_path} \\
    --restart-overhead-hours=0.2 \\
    --deadline-hours=52 \\
    --output-dir exp-converted-{instance_type}-region2 \\
    --task-duration-hours=48

# Multi-region test
python ./main.py --strategy=multi_region_rc_cr_threshold \\
    --env multi_trace \\
    --trace-files {region1_path} {region2_path} \\
    --restart-overhead-hours=0.2 \\
    --deadline-hours=52 \\
    --output-dir exp-converted-{instance_type}-multi \\
    --task-duration-hours=48

"""
        
        script_content += """
echo "All tests completed!"
"""
        
        with open("test_converted_multi_region.sh", "w") as f:
            f.write(script_content)
        
        os.chmod("test_converted_multi_region.sh", 0o755)
        logger.info("Generated test_converted_multi_region.sh")

if __name__ == "__main__":
    converter = DataConverter()
    
    # Convert CSV data to simulator format
    logger.info("Converting CSV availability data to simulator format...")
    converted_regions = converter.convert_all_csv_data()
    
    if converted_regions:
        # Analyze converted data
        analysis_results = converter.analyze_converted_data(converted_regions)
        
        # Suggest region pairs
        suggested_pairs = converter.suggest_region_pairs(analysis_results)
        
        # Generate test script
        if suggested_pairs:
            converter.generate_multi_region_test_script(suggested_pairs)
            
            logger.info(f"\n{'='*60}")
            logger.info("NEXT STEPS:")
            logger.info(f"{'='*60}")
            logger.info("1. Run: python evaluate_multi_region.py  (for existing data)")
            logger.info("2. Run: python convert_data_for_multi_region.py  (to convert more data)")
            logger.info("3. Run: ./test_converted_multi_region.sh  (to test converted data)")
        else:
            logger.warning("No good region pairs found in converted data")
    else:
        logger.error("No data was converted successfully")