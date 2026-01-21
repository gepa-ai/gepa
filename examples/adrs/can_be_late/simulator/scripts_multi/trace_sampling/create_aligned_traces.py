"""
Create aligned trace files from 2023 availability CSV data.

IMPORTANT: This script converts availability data (1=available, 0=unavailable) to 
the preempted format (1=preempted, 0=available) used by TraceEnv, because 
TraceEnv.spot_available() returns "not trace[tick]".
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
SOURCE_CSV_DIR = Path("data/real/availability/2023-02-15/processed")
TARGET_DATA_DIR = Path("data/converted_multi_region_aligned")

def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Loads a CSV and prepares the DataFrame."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def main():
    """Main function to create aligned traces from original CSVs."""
    if not SOURCE_CSV_DIR.exists():
        logging.error(f"Source CSV directory not found: {SOURCE_CSV_DIR}")
        return

    # 1. Load all region dataframes
    region_dfs = {}
    for csv_file in SOURCE_CSV_DIR.glob("*.csv"):
        region_name = csv_file.stem
        logging.info(f"Loading data for {region_name}...")
        try:
            region_dfs[region_name] = load_and_prepare_data(csv_file)
        except Exception as e:
            logging.error(f"Failed to load {csv_file}: {e}")
            continue

    if not region_dfs:
        logging.error("No data loaded. Exiting.")
        return

    # 2. Find the common time window
    common_start = max(df.index.min() for df in region_dfs.values())
    common_end = min(df.index.max() for df in region_dfs.values())

    if common_start >= common_end:
        logging.error("No common time window found across all CSV files. Exiting.")
        return

    logging.info(f"Common time window found: {common_start} to {common_end}")

    # 3. Generate aligned traces
    num_traces = 100  # Number of trace files to generate
    trace_length_hours = 60
    
    # Assuming gap_seconds is consistent, get it from the first dataframe
    first_df = next(iter(region_dfs.values()))
    gap_seconds = int((first_df.index[1] - first_df.index[0]).total_seconds())
    trace_length_ticks = int(trace_length_hours * 3600 / gap_seconds)

    # Align all dataframes to the common window
    aligned_dfs = {name: df[common_start:common_end] for name, df in region_dfs.items()}

    max_start_index = max(0, len(next(iter(aligned_dfs.values()))) - trace_length_ticks)

    for i in range(num_traces):
        logging.info(f"Generating trace set {i}/{num_traces}...")
        start_idx = np.random.randint(0, max_start_index)
        
        # Get the absolute start time for this trace from the first aligned df
        trace_start_time = next(iter(aligned_dfs.values())).index[start_idx]

        for region_name, df in aligned_dfs.items():
            end_idx = min(start_idx + trace_length_ticks, len(df))
            # Convert availability (1=available) to preempted format (1=preempted)
            # because TraceEnv.spot_available() returns "not trace[tick]"
            availability_data = df['availability'].iloc[start_idx:end_idx]
            trace_data = (~availability_data.astype(bool)).astype(int).tolist()

            new_metadata = {
                "gap_seconds": gap_seconds,
                "start_time": trace_start_time.isoformat(),
                "source_file": str(SOURCE_CSV_DIR / f"{region_name}.csv")
            }

            new_content = {
                "metadata": new_metadata,
                "data": trace_data
            }

            target_dir = TARGET_DATA_DIR / region_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / f"{i}.json"

            with open(target_file, 'w') as f:
                json.dump(new_content, f)

    logging.info(f"Successfully generated {num_traces} sets of aligned traces in {TARGET_DATA_DIR}")

if __name__ == "__main__":
    main()