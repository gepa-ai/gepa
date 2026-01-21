"""
Dynamic migration time model based on empirical S3 transfer measurements.

Based on real-world tests conducted across AWS regions and availability zones,
this module provides realistic migration time estimates for ML checkpoints.
"""

import re
from typing import Tuple


def parse_region_info(region_name: str) -> Tuple[str, str, str]:
    """
    Parse region string to extract AWS region, zone, and instance type.
    
    Args:
        region_name: String like 'us-east-1a_v100_1' or 'us-west-2c_k80_1'
        
    Returns:
        Tuple of (region, zone, instance_type)
        e.g., ('us-east-1', 'a', 'v100')
    """
    match = re.match(r'^([a-z]{2}-[a-z]+-\d+)([a-z])_([a-z0-9]+)_(\d+)$', region_name)
    if not match:
        raise ValueError(f"Invalid region format: {region_name}")
    
    region = match.group(1)  # e.g., 'us-east-1'
    zone = match.group(2)    # e.g., 'a'
    instance_type = match.group(3)  # e.g., 'v100'
    
    return region, zone, instance_type


def get_region_relationship(source_region: str, dest_region: str) -> str:
    """
    Determine the relationship between two regions.
    
    Returns one of:
    - 'same_zone': Same region and zone
    - 'cross_az': Same region, different zone  
    - 'cross_region': Different regions in same continent
    - 'cross_continent': Different continents
    """
    src_region, src_zone, _ = parse_region_info(source_region)
    dst_region, dst_zone, _ = parse_region_info(dest_region)
    
    if src_region == dst_region and src_zone == dst_zone:
        return 'same_zone'
    elif src_region == dst_region:
        return 'cross_az'
    else:
        # Extract continent from region (first part before hyphen)
        src_continent = src_region.split('-')[0]
        dst_continent = dst_region.split('-')[0]
        
        if src_continent == dst_continent:
            return 'cross_region'
        else:
            return 'cross_continent'


def get_transfer_speed_mbps(checkpoint_size_gb: float, relationship: str) -> float:
    """
    Get expected transfer speed based on checkpoint size and geographic relationship.
    
    Based on AWS S3 empirical measurements (experiments/s3_transfer_performance.png):
    - Same region (same zone/cross-AZ): 9.72 Gbps
    - Cross region: 8.20 Gbps  
    - Cross continent: 3.59 Gbps
    """
    # Speed based on real AWS S3 measurements (convert Gbps to Mbps)
    speeds_mbps = {
        'same_zone': 9720,      # 9.72 Gbps
        'cross_az': 9720,       # Same as same_zone for S3
        'cross_region': 8200,   # 8.20 Gbps  
        'cross_continent': 3590 # 3.59 Gbps
    }
    
    return speeds_mbps[relationship]


def get_migration_time_hours(source_region: str, dest_region: str, 
                           checkpoint_size_gb: float,
                           instance_startup_hours: float = 0.033) -> float:
    """
    Calculate total migration time including instance startup and data transfer.
    
    Args:
        source_region: Source region string (e.g., 'us-east-1a_v100_1')
        dest_region: Destination region string  
        checkpoint_size_gb: Size of checkpoint in GB
        instance_startup_hours: Time to start new instance (default 2 minutes)
        
    Returns:
        Total migration time in hours
    """
    # Determine geographic relationship
    relationship = get_region_relationship(source_region, dest_region)
    
    # Get transfer speed
    speed_mbps = get_transfer_speed_mbps(checkpoint_size_gb, relationship)
    
    # Calculate transfer time
    # Convert GB to Mb: GB * 8 * 1024 = Mb
    transfer_time_seconds = (checkpoint_size_gb * 8 * 1024) / speed_mbps
    transfer_time_hours = transfer_time_seconds / 3600
    
    # Total time = startup + transfer
    total_hours = instance_startup_hours + transfer_time_hours
    
    return total_hours


def get_transfer_cost_usd(source_region: str, dest_region: str, 
                         checkpoint_size_gb: float) -> float:
    """
    Calculate S3 transfer cost based on AWS pricing.
    
    Costs:
    - Same region (including cross-AZ): $0.00
    - Cross-region (same continent): $0.02/GB
    - Cross-continent: $0.09/GB
    """
    relationship = get_region_relationship(source_region, dest_region)
    
    cost_per_gb = {
        'same_zone': 0.00,
        'cross_az': 0.00,  # AWS doesn't charge for cross-AZ S3 transfers
        'cross_region': 0.02,
        'cross_continent': 0.09
    }
    
    return checkpoint_size_gb * cost_per_gb[relationship]


# For backward compatibility with existing code
def get_fixed_migration_overhead_hours() -> float:
    """Legacy function returning fixed overhead. Use get_migration_time_hours instead."""
    return 0.2  # Original fixed 12-minute overhead


if __name__ == "__main__":
    # Test the module
    test_cases = [
        ("us-east-1a_v100_1", "us-east-1a_v100_1", 100),  # Same zone
        ("us-east-1a_v100_1", "us-east-1c_v100_1", 100),  # Cross-AZ
        ("us-east-1a_v100_1", "us-west-2a_v100_1", 100),  # Cross-region
        ("us-west-2a_v100_1", "eu-west-1a_v100_1", 100),  # Cross-continent (hypothetical)
    ]
    
    print("Migration Time Model Test Results:")
    print("=" * 80)
    
    for src, dst, size_gb in test_cases:
        try:
            relationship = get_region_relationship(src, dst)
            time_hours = get_migration_time_hours(src, dst, size_gb)
            cost_usd = get_transfer_cost_usd(src, dst, size_gb)
            
            print(f"\nSource: {src}")
            print(f"Dest:   {dst}")
            print(f"Size:   {size_gb} GB")
            print(f"Type:   {relationship}")
            print(f"Time:   {time_hours:.2f} hours ({time_hours*60:.1f} minutes)")
            print(f"Cost:   ${cost_usd:.2f}")
        except ValueError as e:
            print(f"\nError processing {src} -> {dst}: {e}")