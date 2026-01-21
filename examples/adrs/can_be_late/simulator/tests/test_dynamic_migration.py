"""Tests for dynamic migration time functionality."""

import json
import os
import tempfile
import unittest
from typing import Dict, Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action
from sky_spot.utils import ClusterType
from sky_spot import task as task_lib
from sky_spot.migration_model import (
    parse_region_info,
    get_region_relationship,
    get_migration_time_hours,
    get_transfer_cost_usd
)


class MockTask(task_lib.Task):
    """Mock task for testing with configurable checkpoint size."""
    
    def __init__(self, duration_seconds: float, checkpoint_size_gb: float = 50.0):
        self.duration_seconds = duration_seconds
        self.checkpoint_size_gb = checkpoint_size_gb
        self._progress_source = []
    
    def set_progress_source(self, task_done_time_list):
        self._progress_source = task_done_time_list
    
    def get_total_duration_seconds(self) -> float:
        return self.duration_seconds
    
    def get_info(self) -> dict:
        progress = sum(self._progress_source)
        return {
            'progress': progress,
            'remaining': self.duration_seconds - progress,
            'total': self.duration_seconds
        }
    
    @property
    def is_done(self) -> bool:
        return sum(self._progress_source) >= self.duration_seconds
    
    def get_config(self) -> dict:
        return {
            'duration_seconds': self.duration_seconds,
            'checkpoint_size_gb': self.checkpoint_size_gb
        }
    
    def reset(self):
        """Reset the task state."""
        self._progress_source = []
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"MockTask(duration={self.duration_seconds}s, checkpoint={self.checkpoint_size_gb}GB)"


class MockArgs:
    """Mock arguments for strategy initialization."""
    def __init__(self, restart_overhead_hours=0.2):
        self.deadline_hours = 100.0  # Large deadline to avoid SAFETY NET
        self.restart_overhead_hours = [restart_overhead_hours]  # Fixed overhead (will be overridden for migrations)
        self.inter_task_overhead = [0.0]


class TestDynamicMigration(unittest.TestCase):
    """Test dynamic migration time functionality."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 600  # 10 minutes per tick
        
        # Create region-specific directories
        self.regions = {
            'us-east-1a_v100_1': {'region': 'us-east-1', 'zone': 'a'},
            'us-east-1c_v100_1': {'region': 'us-east-1', 'zone': 'c'},
            'us-west-2a_v100_1': {'region': 'us-west-2', 'zone': 'a'},
        }
        
        self.trace_files = []
        for region_name, info in self.regions.items():
            # Create directory structure
            region_dir = os.path.join(self.temp_dir, region_name)
            os.makedirs(region_dir, exist_ok=True)
            
            # Create trace file
            trace_file = os.path.join(region_dir, '0.json')
            self.create_trace_file(trace_file, region_name, [0] * 100)  # All available
            self.trace_files.append(trace_file)
    
    def create_trace_file(self, filepath: str, region_name: str, availability: list):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "region": region_name,
                "instance_type": "v100_1",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            "data": availability
        }
        
        with open(filepath, 'w') as f:
            json.dump(trace_data, f)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_migration_model_parsing(self):
        """Test region parsing and relationship detection."""
        # Test parsing
        region, zone, instance_type = parse_region_info('us-east-1a_v100_1')
        self.assertEqual(region, 'us-east-1')
        self.assertEqual(zone, 'a')
        self.assertEqual(instance_type, 'v100')
        
        # Test relationships
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-east-1a_v100_1'),
            'same_zone'
        )
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-east-1c_v100_1'),
            'cross_az'
        )
        self.assertEqual(
            get_region_relationship('us-east-1a_v100_1', 'us-west-2a_v100_1'),
            'cross_region'
        )
    
    def test_migration_time_calculation(self):
        """Test migration time calculations for different scenarios."""
        checkpoint_size = 100  # GB
        
        # Same zone: should be fast (startup time + transfer at 9.72 Gbps)
        same_zone_hours = get_migration_time_hours(
            'us-east-1a_v100_1', 'us-east-1a_v100_1', checkpoint_size
        )
        # Expected: 0.033 (startup) + 100GB at 9720 Mbps = 0.033 + 0.023 = 0.056 hours
        self.assertAlmostEqual(same_zone_hours, 0.056, places=3)
        
        # Cross-AZ: should be same as same zone for S3
        cross_az_hours = get_migration_time_hours(
            'us-east-1a_v100_1', 'us-east-1c_v100_1', checkpoint_size
        )
        # Expected: 0.033 (startup) + 100GB at 9720 Mbps = 0.033 + 0.023 = 0.056 hours
        self.assertAlmostEqual(cross_az_hours, 0.056, places=3)
        
        # Cross-region: should be slightly slower
        cross_region_hours = get_migration_time_hours(
            'us-east-1a_v100_1', 'us-west-2a_v100_1', checkpoint_size
        )
        # Expected: 0.033 (startup) + 100GB at 8200 Mbps = 0.033 + 0.028 = 0.061 hours
        self.assertAlmostEqual(cross_region_hours, 0.061, places=3)
    
    def test_cold_start_vs_migration(self):
        """Test that cold starts use fixed overhead while migrations use dynamic time."""
        env = MultiTraceEnv(self.trace_files[:2], env_start_hours=0)  # us-east-1a and us-east-1c
        
        class MigrationStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Cold start in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 5:
                    # Migrate to region 1 (cross-AZ)
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        args = MockArgs(restart_overhead_hours=0.2)  # 12 minutes fixed overhead
        task = MockTask(duration_seconds=36000, checkpoint_size_gb=100)  # 10 hour task, 100GB checkpoint
        strategy = MigrationStrategy(args)
        strategy.reset(env, task)
        
        progress_history = []
        overhead_history = []
        
        # Run for several ticks
        for i in range(10):
            env.observe()
            
            # Record overhead BEFORE update_strategy_progress modifies it
            overhead_before_update = strategy.remaining_restart_overhead
            
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1
            
            # Record progress and overhead
            if strategy.task_done_time:
                progress_history.append(strategy.task_done_time[-1])
            else:
                progress_history.append(0)
            overhead_history.append(strategy.remaining_restart_overhead)
            
        
        # Analyze results
        # Tick 0: Launch happens, but overhead not yet applied
        self.assertEqual(progress_history[0], 0)  # No progress yet
        self.assertEqual(overhead_history[0], 0)  # Overhead not yet applied
        
        # Tick 1: Overhead is applied (0.2 hours = 720 seconds)
        # Since gap_seconds = 600, only 600 seconds of overhead consumed
        self.assertEqual(progress_history[1], 0)  # No progress during overhead
        self.assertAlmostEqual(overhead_history[1], 720 - 600, delta=1)  # 120 seconds remaining
        
        # Tick 2: Finish cold start overhead
        self.assertGreater(progress_history[2], 0)  # Should have some progress
        self.assertAlmostEqual(overhead_history[2], 0, delta=1)  # Overhead complete
        
        # Ticks 3-4: Normal progress  
        for i in range(3, 5):
            self.assertAlmostEqual(progress_history[i], 600, delta=1)  # Full tick progress
        
        # Tick 5: Migration starts - but overhead not applied until next tick
        self.assertEqual(overhead_history[5], 0)  # Not yet applied
        
        # Tick 6: Dynamic migration overhead should be applied
        # Cross-AZ migration for 100GB with 0.2 hour startup time
        # Expected: 0.2 (startup) + 0.228 (transfer) = 0.428 hours = 1540.8 seconds
        expected_migration_hours = get_migration_time_hours(
            'us-east-1a_v100_1', 'us-east-1c_v100_1', 100,
            instance_startup_hours=0.2  # Match the restart_overhead_hours
        )
        expected_migration_seconds = expected_migration_hours * 3600
        
        # Since gap_seconds = 600, remaining overhead after tick 6 should be 1540.8 - 600 = 940.8
        self.assertEqual(progress_history[6], 0)  # No progress during overhead
        self.assertAlmostEqual(overhead_history[6], expected_migration_seconds - 600, delta=10)
        
        # The overhead should be different from the fixed overhead
        self.assertNotAlmostEqual(overhead_history[6], 120, delta=50)  # Not the fixed overhead remainder
    
    def test_multiple_migrations(self):
        """Test multiple migrations with different relationships."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)  # All 3 regions
        
        class MultiMigrationStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Start in us-east-1a
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 3:
                    # Migrate to us-east-1c (cross-AZ)
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 6:
                    # Migrate to us-west-2a (cross-region)
                    yield Terminate(region=1)
                    result = yield TryLaunch(region=2, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        args = MockArgs(restart_overhead_hours=0.2)
        task = MockTask(duration_seconds=36000, checkpoint_size_gb=50)  # 50GB checkpoint
        strategy = MultiMigrationStrategy(args)
        strategy.reset(env, task)
        
        migration_overheads = []
        migration_ticks = [3, 6]  # When migrations happen
        
        # Run simulation
        for i in range(10):
            env.observe()
            env.update_strategy_progress(strategy)
            
            # Record overhead at migration ticks
            if i in migration_ticks:
                migration_overheads.append({
                    'tick': i,
                    'overhead_before': strategy.remaining_restart_overhead
                })
            
            env.execute_multi_strategy(strategy)
            env.tick += 1
            
            # Record overhead after execution
            if i in migration_ticks:
                migration_overheads[-1]['overhead_after'] = strategy.remaining_restart_overhead
        
        # Verify different migration times
        # First migration: cross-AZ (us-east-1a -> us-east-1c)
        cross_az_hours = get_migration_time_hours(
            'us-east-1a_v100_1', 'us-east-1c_v100_1', 50
        )
        cross_az_seconds = cross_az_hours * 3600
        
        # Second migration: cross-region (us-east-1c -> us-west-2a)
        cross_region_hours = get_migration_time_hours(
            'us-east-1c_v100_1', 'us-west-2a_v100_1', 50
        )
        cross_region_seconds = cross_region_hours * 3600
        
        # Cross-region should take longer than cross-AZ (though difference is smaller with fast S3)
        self.assertGreater(cross_region_seconds, cross_az_seconds)
    
    def test_migration_costs(self):
        """Test that migration costs are calculated correctly."""
        # Test various scenarios
        test_cases = [
            ('us-east-1a_v100_1', 'us-east-1a_v100_1', 100, 0.0),  # Same zone: free
            ('us-east-1a_v100_1', 'us-east-1c_v100_1', 100, 0.0),  # Cross-AZ: free
            ('us-east-1a_v100_1', 'us-west-2a_v100_1', 100, 2.0),  # Cross-region: $0.02/GB
        ]
        
        for src, dst, size_gb, expected_cost in test_cases:
            cost = get_transfer_cost_usd(src, dst, size_gb)
            self.assertEqual(cost, expected_cost)
    
    def test_checkpoint_size_impact(self):
        """Test that different checkpoint sizes result in different migration times."""
        source = 'us-east-1a_v100_1' 
        dest = 'us-west-2a_v100_1'  # Cross-region for maximum impact
        startup_hours = 0.033
        
        checkpoint_sizes = [10, 50, 100, 500]
        migration_times = []
        
        for size in checkpoint_sizes:
            time_hours = get_migration_time_hours(source, dest, size, startup_hours)
            migration_times.append(time_hours)
            print(f"Checkpoint {size}GB: {time_hours:.3f} hours ({time_hours*60:.1f} minutes)")
        
        # Verify that larger checkpoints take longer
        for i in range(1, len(migration_times)):
            self.assertGreater(migration_times[i], migration_times[i-1],
                             f"Checkpoint {checkpoint_sizes[i]}GB should take longer than {checkpoint_sizes[i-1]}GB")
        
        # Verify the progression makes sense
        # With high-speed S3 transfer, the difference is smaller but still significant
        # 10GB -> 500GB should be at least 3x longer
        self.assertGreater(migration_times[-1] / migration_times[0], 3,
                          "500GB should take noticeably longer than 10GB")
    
    def test_checkpoint_size_in_environment(self):
        """Test that checkpoint size affects migration overhead in the environment."""
        
        class CheckpointTestStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Start in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 3:
                    # Migrate to region 1 (cross-AZ)
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        # Test with different checkpoint sizes
        results = {}
        for checkpoint_size in [10, 100, 500]:
            # Create fresh environment for each test
            env = MultiTraceEnv(self.trace_files[:2], env_start_hours=0)
            
            args = MockArgs(restart_overhead_hours=0.2)
            task = MockTask(duration_seconds=36000, checkpoint_size_gb=checkpoint_size)
            strategy = CheckpointTestStrategy(args)
            strategy.reset(env, task)
            
            # Run until migration happens
            migration_overhead = None
            for i in range(8):
                env.observe()
                env.update_strategy_progress(strategy)
                
                # Check overhead after migration (at tick 4)
                if i == 4:  # One tick after migration
                    migration_overhead = strategy.remaining_restart_overhead
                
                env.execute_multi_strategy(strategy)
                env.tick += 1
            
            results[checkpoint_size] = migration_overhead
            print(f"Checkpoint {checkpoint_size}GB migration overhead: {migration_overhead:.1f} seconds")
        
        # Verify that larger checkpoints result in larger overhead
        self.assertGreater(results[100], results[10],
                          "100GB checkpoint should have more overhead than 10GB")
        self.assertGreater(results[500], results[100], 
                          "500GB checkpoint should have more overhead than 100GB")


if __name__ == '__main__':
    unittest.main()