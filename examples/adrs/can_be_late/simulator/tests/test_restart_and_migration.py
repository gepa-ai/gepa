"""Tests for restart overhead and migration in multi-region environments."""

import json
import os
import tempfile
import unittest
from typing import Dict, Optional, Generator

from sky_spot.env import Env, MultiTraceEnv, TraceEnv
from sky_spot.strategies.strategy import Strategy, MultiRegionStrategy
from sky_spot.multi_region_types import TryLaunch, Terminate, LaunchResult, Action
from sky_spot.utils import ClusterType
from sky_spot.trace import Trace
from sky_spot import task as task_lib


class MockTask(task_lib.Task):
    """Mock task for testing."""
    
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self._progress_source = []
    
    def set_progress_source(self, progress_source):
        self._progress_source = progress_source
    
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
        return {'duration_seconds': self.duration_seconds}
    
    def reset(self):
        """Reset the task state."""
        self._progress_source = []
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"MockTask(duration={self.duration_seconds}s)"


class AlwaysSpotStrategy(Strategy):
    """Strategy that always requests SPOT instances."""
    NAME = 'always_spot_test'
    
    def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
        return ClusterType.SPOT
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


class AlwaysSpotMultiRegionStrategy(MultiRegionStrategy):
    """Multi-region strategy that always requests SPOT instances."""
    NAME = 'always_spot_multi_test'
    
    def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
        while not self.task.is_done:
            # Try region 0 first
            result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
            if result.success:
                continue
            
            # If region 0 fails, try region 1
            result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
    
    @classmethod
    def _from_args(cls, parser):
        args, _ = parser.parse_known_args()
        return cls(args)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality before testing complex scenarios."""
    
    def test_spot_availability(self):
        """Test that spot availability is correctly reported."""
        # Create a trace with spot availability pattern
        # 0 = available, 1 = preempted/unavailable
        trace_data = {
            'metadata': {
                'gap_seconds': 600,
                'region': 'us-east-1',
                'zone': 'us-east-1a',
                'instance_type': 'p3.2xlarge'
            },
            'data': [0, 1, 0]  # available, unavailable, available
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_v100_1.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_file = f.name
        
        try:
            env = TraceEnv(trace_file=trace_file, env_start_hours=0)
            env.reset()
            
            # Test spot availability at each tick
            # Tick 0: data[0]=0, so spot should be available
            self.assertTrue(env.spot_available(), "Spot should be available at tick 0")
            
            env.tick = 1
            # Tick 1: data[1]=1, so spot should NOT be available
            self.assertFalse(env.spot_available(), "Spot should NOT be available at tick 1")
            
            env.tick = 2
            # Tick 2: data[2]=0, so spot should be available
            self.assertTrue(env.spot_available(), "Spot should be available at tick 2")
            
        finally:
            os.unlink(trace_file)


class TestRestartOverhead(unittest.TestCase):
    """Test restart overhead (cold start) behavior."""
    
    def setUp(self):
        # Create mock arguments
        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.3]  # 18 minutes = 1080 seconds
            inter_task_overhead = [0.0]
            gap_seconds = 600  # 10 minutes
            strategy = 'always_spot_test'
        
        self.args = Args()
        self.task = MockTask(3600)  # 1 hour task
    
    def test_cold_start_overhead_single_region(self):
        """Test that cold start overhead is correctly applied in single-region."""
        # Create a trace with constant spot availability
        # 0 = available, 1 = preempted
        trace_data = {
            'metadata': {
                'gap_seconds': 600,
                'region': 'us-east-1', 
                'zone': 'us-east-1a',
                'instance_type': 'p3.2xlarge'
            },
            'data': [0] * 10  # All available
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_v100_1.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_file = f.name
        
        try:
            env = TraceEnv(trace_file=trace_file, env_start_hours=0)
            strategy = AlwaysSpotStrategy(self.args)
            
            # Run simulation
            env.reset()
            strategy.reset(env, self.task)
            
            per_tick_progress = []
            while not self.task.is_done and env.tick < 10:
                cluster_type = strategy.step()
                env.step(cluster_type)
                # Get the progress made in this tick (last element of task_done_time)
                if hasattr(strategy, 'task_done_time') and strategy.task_done_time:
                    per_tick_progress.append(strategy.task_done_time[-1])
            
            # Check total progress equals task duration
            actual_sum = sum(per_tick_progress)
            self.assertEqual(actual_sum, 3600, f"Total progress should be 3600s, got {actual_sum}s")
            
            # Check restart overhead is applied
            self.assertEqual(per_tick_progress[0], 0, "First tick should have 0 progress (cold start)")
            self.assertEqual(per_tick_progress[1], 0, "Second tick should have 0 progress (still in overhead)")
            # Third tick should have partial progress
            self.assertGreater(per_tick_progress[2], 0, "Third tick should have some progress")
            self.assertLess(per_tick_progress[2], 600, "Third tick should have partial progress due to overhead")
            
        finally:
            os.unlink(trace_file)
    
    def test_cold_start_overhead_multi_region(self):
        """Test that cold start overhead is correctly applied in multi-region."""
        # Create traces for 2 regions
        trace_files = []
        for region in range(2):
            trace_data = {
                'metadata': {
                    'gap_seconds': 600,
                    'region': f'us-east-{region+1}',
                    'zone': f'us-east-{region+1}a',
                    'instance_type': 'p3.2xlarge'
                },
                'data': [0] * 10  # All available
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_region{region}_v100_1.json', delete=False) as f:
                json.dump(trace_data, f)
                trace_files.append(f.name)
        
        try:
            env = MultiTraceEnv(trace_files=trace_files, env_start_hours=0)
            strategy = AlwaysSpotMultiRegionStrategy(self.args)
            
            # Run simulation
            env.reset()
            strategy.reset(env, self.task)
            
            per_tick_progress = []
            
            # Launch instance at the beginning  
            success = env._try_launch_internal(0, ClusterType.SPOT)
            self.assertTrue(success, "Initial launch should succeed")
            
            # Simulate ticks
            for tick in range(10):
                if not self.task.is_done:
                    # Observe current state
                    env.observe()
                    
                    # Update strategy progress
                    env.update_strategy_progress(strategy)
                    
                    # Record progress
                    if hasattr(strategy, 'task_done_time') and strategy.task_done_time:
                        per_tick_progress.append(strategy.task_done_time[-1])
                    
                    # Step to next tick
                    env.tick += 1
            
            # Check total progress
            actual_sum = sum(per_tick_progress)
            self.assertEqual(actual_sum, 3600, f"Total progress should be 3600s, got {actual_sum}s")
            
            # We should have at least one tick with zero progress (cold start)
            zero_progress_ticks = sum(1 for p in per_tick_progress if p == 0)
            self.assertGreaterEqual(zero_progress_ticks, 1, "Should have at least one tick with zero progress")
            
        finally:
            for f in trace_files:
                os.unlink(f)


class TestMigration(unittest.TestCase):
    """Test migration scenarios between regions."""
    
    def setUp(self):
        # Create mock arguments
        class Args:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]  # 12 minutes = 720 seconds
            inter_task_overhead = [0.0]
            gap_seconds = 600  # 10 minutes
            strategy = 'migration_test'
        
        self.args = Args()
        self.task = MockTask(3600)  # 1 hour task
    
    def test_migration_restart_overhead(self):
        """Test that migration between regions applies restart overhead."""
        # Create traces where region 0 fails after 3 ticks
        trace_files = []
        
        # Region 0: available for first 3 ticks, then unavailable
        # 0 = available, 1 = preempted
        trace_data_0 = {
            'metadata': {
                'gap_seconds': 600,
                'region': 'us-east-1',
                'zone': 'us-east-1a',
                'instance_type': 'p3.2xlarge'
            },
            'data': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # Available first 3 ticks
        }
        
        # Region 1: always available
        trace_data_1 = {
            'metadata': {
                'gap_seconds': 600,
                'region': 'us-east-2',
                'zone': 'us-east-2a',
                'instance_type': 'p3.2xlarge'
            },
            'data': [0] * 10  # Always available
        }
        
        # Create proper directory structure for region parsing
        import tempfile
        import os
        temp_base = tempfile.mkdtemp()
        
        # Region 0: us-east-1a
        region0_dir = os.path.join(temp_base, 'us-east-1a_v100_1')
        os.makedirs(region0_dir, exist_ok=True)
        region0_file = os.path.join(region0_dir, '0.json')
        with open(region0_file, 'w') as f:
            json.dump(trace_data_0, f)
        trace_files.append(region0_file)
        
        # Region 1: us-east-2a  
        region1_dir = os.path.join(temp_base, 'us-east-2a_v100_1')
        os.makedirs(region1_dir, exist_ok=True)
        region1_file = os.path.join(region1_dir, '0.json')
        with open(region1_file, 'w') as f:
            json.dump(trace_data_1, f)
        trace_files.append(region1_file)
        
        try:
            env = MultiTraceEnv(trace_files=trace_files, env_start_hours=0)
            
            # Custom migration strategy
            class MigrationTestStrategy(MultiRegionStrategy):
                NAME = 'migration_test'
                
                def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                    current_region = 0
                    
                    while not self.task.is_done:
                        # Try current region
                        result = yield TryLaunch(region=current_region, cluster_type=ClusterType.SPOT)
                        
                        if not result.success and current_region == 0:
                            # Migrate to region 1
                            current_region = 1
                            continue
                
                @classmethod
                def _from_args(cls, parser):
                    args, _ = parser.parse_known_args()
                    return cls(args)
            
            strategy = MigrationTestStrategy(self.args)
            
            # Run simulation
            env.reset()
            strategy.reset(env, self.task)
            
            progress_history = []
            region_history = []
            
            # Launch in region 0 initially
            success = env._try_launch_internal(0, ClusterType.SPOT)
            self.assertTrue(success, "Initial launch in region 0 should succeed")
            region_history.append((0, 0, 'launch'))
            
            # Simulate ticks
            for tick in range(10):
                if not self.task.is_done:
                    # Observe current state (handles preemptions)
                    env.observe()
                    
                    # Check if region 0 was preempted and we need to migrate
                    if tick == 3 and 0 not in env.get_active_instances():
                        # Region 0 spot became unavailable, migrate to region 1
                        region_history.append((tick, 0, 'preempt'))
                        success = env._try_launch_internal(1, ClusterType.SPOT)
                        self.assertTrue(success, "Migration to region 1 should succeed")
                        region_history.append((tick, 1, 'launch'))
                    
                    # Update strategy progress
                    env.update_strategy_progress(strategy)
                    
                    # Record progress
                    if hasattr(strategy, 'task_done_time') and strategy.task_done_time:
                        progress = strategy.task_done_time[-1]
                        progress_history.append(progress)
                        print(f"Tick {tick}: progress={progress}, "
                              f"active_regions={list(env.get_active_instances().keys())}")
                    
                    # Step to next tick
                    env.tick += 1
            
            print(f"\nProgress history: {progress_history}")
            print(f"Region history: {region_history}")
            
            # Verify migration occurred
            launches = [(t, r) for t, r, action in region_history if action == 'launch']
            self.assertEqual(len(launches), 2, "Should have two launches (initial + migration)")
            self.assertEqual(launches[0][1], 0, "First launch should be in region 0")
            self.assertEqual(launches[1][1], 1, "Second launch should be in region 1")
            
            # Verify restart overhead was applied after migration
            # Find where migration occurred
            migration_tick = launches[1][0]
            
            # Progress at migration tick might still be from the old instance
            # Check the NEXT tick after migration for restart overhead effect
            print(f"Migration occurred at tick {migration_tick}")
            
            # The restart overhead should show up in the tick after migration
            if migration_tick + 1 < len(progress_history):
                print(f"Progress at tick {migration_tick + 1} (after migration): {progress_history[migration_tick + 1]}")
                # After migration, the next tick should show restart overhead effect
                # Either 0 progress or reduced progress
                self.assertLessEqual(progress_history[migration_tick + 1], 120,
                               "Tick after migration should have minimal progress due to restart overhead")
            
            # Verify total task completion
            total_progress = sum(progress_history)
            self.assertEqual(total_progress, 3600, f"Total progress should be 3600s, got {total_progress}s")
            
        finally:
            for f in trace_files:
                os.unlink(f)


if __name__ == '__main__':
    unittest.main() 