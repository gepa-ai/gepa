# Cloudcast Broadcast Optimization

This example demonstrates using GEPA's `optimize_anything` API to optimize a broadcast routing algorithm for multi-cloud data transfer.

## Requirements

Install the required dependencies:

```bash
pip install networkx pandas numpy

# Optional (for colored output)
pip install colorama
```

Or install from the requirements file:

```bash
pip install -r examples/adrs/cloudcast/requirements.txt
```

## Testing the Setup

Before running the optimization, verify everything is working:

```bash
python examples/adrs/cloudcast/test_evaluator.py
```

This will test all components and report any issues.

## Problem Description

The Cloudcast problem involves efficiently transferring data from a single source to multiple destinations across different cloud providers (AWS, GCP, Azure). The algorithm must:

- Find paths from a source region to all destination regions
- Minimize total cost (egress fees + instance running costs)
- Support data partitioning for parallel transfers
- Handle heterogeneous bandwidth and pricing between cloud providers

## Directory Structure

```
cloudcast/
├── main.py              # Entry point using optimize_anything API
├── evaluator.py         # Fitness function for GEPA
├── __init__.py          # Module exports
├── README.md            # This file
└── cloudcast/           # Core cloudcast modules
    ├── __init__.py
    ├── broadcast.py     # BroadCastTopology data structure
    ├── simulator.py     # BCSimulator for path evaluation
    ├── utils.py         # Graph utilities
    ├── initial_program.py  # Baseline search algorithm
    ├── config/          # Configuration files
    │   ├── intra_aws.json
    │   ├── intra_azure.json
    │   ├── intra_gcp.json
    │   ├── inter_agz.json
    │   └── inter_gaz2.json
    └── profiles/        # Network profiles
        ├── cost.csv     # Egress costs between regions
        └── throughput.csv  # Bandwidth between regions
```

## Usage

Run the optimization:

```bash
# From the repository root
python examples/adrs/cloudcast/main.py --model gpt-4o-mini

# With custom settings
python examples/adrs/cloudcast/main.py \
    --model gpt-4o-mini \
    --max-metric-calls 50 \
    --minibatch-size 3 \
    --run-dir ./runs/cloudcast_test
```

### Command Line Arguments

- `--model`: (Required) LLM model for reflection (e.g., "gpt-4o-mini", "gemini-1.5-flash")
- `--max-metric-calls`: Maximum fitness evaluations (default: 100)
- `--minibatch-size`: Samples per reflection batch (default: 3)
- `--config-dir`: Path to configuration files directory
- `--run-dir`: Output directory for results

## Configuration Files

Each configuration file specifies a broadcast scenario:

```json
{
    "source_node": "aws:ap-northeast-1",
    "dest_nodes": ["aws:me-south-1", "aws:eu-south-1", ...],
    "data_vol": 300,
    "num_partitions": 10,
    "ingress_limit": {"aws": 10, "gcp": 16, "azure": 16},
    "egress_limit": {"aws": 5, "gcp": 7, "azure": 16}
}
```

## Algorithm Interface

The evolved algorithm must implement:

```python
def search_algorithm(src, dsts, G, num_partitions):
    """
    Find broadcast paths from source to all destinations.
    
    Args:
        src: Source node (e.g., "aws:us-east-1")
        dsts: List of destination nodes
        G: NetworkX DiGraph with 'cost' and 'throughput' edge attributes
        num_partitions: Number of data partitions
        
    Returns:
        BroadCastTopology with paths for all (destination, partition) pairs
    """
```

## Scoring

- **Score**: `1 / (1 + total_cost)` (higher is better)
- **Total Cost**: Egress costs + Instance costs
- **Egress Cost**: Sum of (data_volume × edge_cost) for all edges used
- **Instance Cost**: Number of nodes × instance_hourly_rate × transfer_time

## Output

Results are saved to the run directory:

- `best_program.py`: The optimized search algorithm
- `metrics.json`: Performance comparison (baseline vs optimized)
- `candidates/`: All candidate programs generated during optimization
