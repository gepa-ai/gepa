"""Tests for MultiTraceEnv bugs we discovered and fixed."""

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


class TestRestartOverheadBugs(unittest.TestCase):
    """Test restart overhead application logic."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 195  # Standard gap seconds
        
        # Create simple trace - always available
        self.create_trace_file(0, [0] * 200)  # Region 0
        self.create_trace_file(1, [0] * 200)  # Region 1
        
        self.trace_files = [
            os.path.join(self.temp_dir, f"us-east-{i}a_v100_1", "0.json")
            for i in range(2)
        ]
    
    def create_trace_file(self, region_id: int, availability: list):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "region": f"us-east-{region_id}",
                "zone": f"us-east-{region_id}a",
                "instance_type": "v100",
                "device": "v100_1",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds
            },
            "data": availability
        }
        
        # Create proper directory structure
        region_dir = os.path.join(self.temp_dir, f"us-east-{region_id}a_v100_1")
        os.makedirs(region_dir, exist_ok=True)
        filename = os.path.join(region_dir, "0.json")
        with open(filename, 'w') as f:
            json.dump(trace_data, f)
    
    def test_restart_overhead_not_applied_when_running(self):
        """Test that restart overhead is NOT applied when instance is already running."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TestStrategy(MultiRegionStrategy):
            """Strategy that launches once and then does nothing."""
            NAME = 'test_no_spurious_overhead'
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Only launch on first call
                if not hasattr(self, '_launched'):
                    self._launched = False
                
                if not self._launched:
                    # Launch in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                    self._launched = True
                
                # Now just run without any actions for many ticks
                # The bug would cause restart overhead to be applied incorrectly
        
        # Create strategy with standard restart overhead
        class MockArgs:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]  # 720 seconds
            inter_task_overhead = [0.0]
        
        strategy = TestStrategy(MockArgs())
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy.reset(env, task)  # Properly initialize strategy with env and task
        
        # Execute for several ticks
        for tick in range(10):
            env.observe()
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # After initial restart overhead (ticks 1-4), should have no more overhead
        # Tick 1-3: 0 progress (720 seconds of overhead)
        # Tick 4: 60 seconds progress (195 - 135 remaining overhead)
        # Tick 5+: 195 seconds progress each
        
        # Check that we don't have spurious restart overhead at later ticks
        self.assertEqual(strategy.task_done_time[0], 0)  # Tick 0: no previous tick
        self.assertEqual(strategy.task_done_time[1], 0)  # Tick 1: full overhead
        self.assertEqual(strategy.task_done_time[2], 0)  # Tick 2: still overhead
        self.assertEqual(strategy.task_done_time[3], 0)  # Tick 3: still overhead
        self.assertEqual(strategy.task_done_time[4], 60)  # Tick 4: partial progress
        
        # These should all be 195 (full tick progress) - the bug would make some 0
        for i in range(5, 10):
            self.assertEqual(strategy.task_done_time[i], 195, 
                           f"Tick {i} should have full progress, not restart overhead")
    
    def test_restart_overhead_applied_on_same_region_restart(self):
        """Test that restart overhead IS applied when restarting in the same region."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TestStrategy(MultiRegionStrategy):
            """Strategy that terminates and relaunches in same region."""
            NAME = 'test_same_region_restart'
            
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.tick_count == 0:
                    # Launch in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self.tick_count == 5:
                    # Terminate
                    yield Terminate(region=0)
                elif self.tick_count == 6:
                    # Relaunch in SAME region
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self.tick_count += 1
        
        # Create strategy with standard restart overhead
        class MockArgs:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]  # 720 seconds
            inter_task_overhead = [0.0]
        
        strategy = TestStrategy(MockArgs())
        task = MockTask(duration_seconds=7200)  # 2 hour task
        strategy.reset(env, task)  # Properly initialize strategy with env and task
        
        # Execute for 15 ticks to cover both launches
        for tick in range(15):
            env.observe()
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # Find where the restart happens (around tick 7-8)
        # Should see restart overhead applied TWICE:
        # 1. Initial launch (ticks 1-4)
        # 2. After terminate and relaunch (around ticks 8-11)
        
        # Count ticks with 0 progress (indicating overhead)
        zero_progress_ticks = [i for i, p in enumerate(strategy.task_done_time) if p == 0]
        
        # Should have at least 6 ticks with 0 progress (3 for each restart)
        self.assertGreaterEqual(len(zero_progress_ticks), 6,
                               "Should have restart overhead for both launches")


class TestInstanceSwitching(unittest.TestCase):
    """Test instance type switching within same region."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 195
        
        # Create traces
        self.create_trace_file(0, [0] * 100)  # Always available
        self.trace_files = [
            os.path.join(self.temp_dir, f"us-east-{i}a_v100_1", "0.json")
            for i in range(1)
        ]
    
    def create_trace_file(self, region_id: int, availability: list):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "region": f"us-east-{region_id}",
                "zone": f"us-east-{region_id}a",
                "instance_type": "v100",
                "device": "v100_1",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds
            },
            "data": availability
        }
        
        # Create proper directory structure
        region_dir = os.path.join(self.temp_dir, f"us-east-{region_id}a_v100_1")
        os.makedirs(region_dir, exist_ok=True)
        filename = os.path.join(region_dir, "0.json")
        with open(filename, 'w') as f:
            json.dump(trace_data, f)
    
    def test_same_tick_terminate_and_launch(self):
        """Test that strategy can terminate and launch in same tick."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TestStrategy(MultiRegionStrategy):
            """Strategy that switches from ON_DEMAND to SPOT in same tick."""
            NAME = 'test_same_tick_switch'
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if not hasattr(self, '_tick_count'):
                    self._tick_count = 0
                
                if self._tick_count == 0:
                    # Launch ON_DEMAND first
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result is not None and result.success
                elif self._tick_count == 2:
                    # Now switch: launch SPOT then terminate ON_DEMAND in same tick
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                    
                    yield Terminate(region=0)  # This should terminate the ON_DEMAND
                
                self._tick_count += 1
        
        class MockArgs:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]
            inter_task_overhead = [0.0]
        
        strategy = TestStrategy(MockArgs())
        task = MockTask(duration_seconds=1800)
        strategy.reset(env, task)  # Properly initialize strategy with env and task
        
        # Execute for several ticks
        for tick in range(5):
            env.observe()
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # After the switch, should only have SPOT running
        active = env.get_active_instances()
        self.assertEqual(len(active), 1, "Should have exactly one instance")
        self.assertEqual(active.get(0), ClusterType.SPOT, "Should be SPOT instance")
    
    def test_launch_without_terminate_fails(self):
        """Test that launching without terminating existing instance fails validation."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class BadStrategy(MultiRegionStrategy):
            """Strategy that tries to launch without terminating."""
            NAME = 'test_bad_strategy'
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if not hasattr(self, '_tick_count'):
                    self._tick_count = 0
                
                if self._tick_count == 0:
                    # Launch ON_DEMAND
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.ON_DEMAND)
                    assert result is not None and result.success
                elif self._tick_count == 1:
                    # Try to launch SPOT without terminating ON_DEMAND
                    # This should fail validation
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    # Strategy ends without terminating - should raise error
                
                self._tick_count += 1
        
        class MockArgs:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]
            inter_task_overhead = [0.0]
        
        strategy = BadStrategy(MockArgs())
        task = MockTask(duration_seconds=1800)
        strategy.reset(env, task)  # Properly initialize strategy with env and task
        
        # First tick: launch ON_DEMAND
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Second tick: try to launch SPOT without terminating (should raise error)
        env.observe()
        with self.assertRaises(ValueError) as context:
            env.execute_multi_strategy(strategy)
        
        self.assertIn("must explicitly terminate", str(context.exception))


class TestMigrationOverhead(unittest.TestCase):
    """Test migration overhead between regions."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 195
        
        # Create traces for different regions
        # Use actual region names that have migration data
        self.regions = ['us-east-1a_v100_1', 'us-west-2a_v100_1']
        self.trace_files = []
        
        for i, region in enumerate(self.regions):
            trace_data = {
                "metadata": {
                    "region": region.split('_')[0],
                    "instance_type": "v100",
                    "start_time": "2024-01-01T00:00:00Z",
                    "gap_seconds": self.gap_seconds,
                    "device": "v100_1"
                },
                "data": [0] * 100  # Always available
            }
            
            # Create directory structure
            region_dir = os.path.join(self.temp_dir, region)
            os.makedirs(region_dir, exist_ok=True)
            filename = os.path.join(region_dir, "0.json")
            
            with open(filename, 'w') as f:
                json.dump(trace_data, f)
            
            self.trace_files.append(filename)
    
    def test_migration_overhead_applied(self):
        """Test that migration overhead is properly applied when switching regions."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TestStrategy(MultiRegionStrategy):
            """Strategy that migrates between regions."""
            NAME = 'test_migration'
            
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if not hasattr(self, '_tick_count'):
                    self._tick_count = 0
                
                if self._tick_count == 0:
                    # Launch in region 0
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                elif self._tick_count == 4:
                    # Terminate and switch to region 1
                    yield Terminate(region=0)
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    assert result is not None and result.success
                
                self._tick_count += 1
        
        class MockArgs:
            deadline_hours = 10.0
            restart_overhead_hours = [0.2]  # 720 seconds
            inter_task_overhead = [0.0]
            checkpoint_size_gb = 50.0  # This affects migration time
        
        strategy = TestStrategy(MockArgs())
        task = MockTask(duration_seconds=7200)
        strategy.reset(env, task)  # Properly initialize strategy with env and task
        
        # Execute for several ticks
        for tick in range(15):
            env.observe()
            env.update_strategy_progress(strategy)
            env.execute_multi_strategy(strategy)
            env.tick += 1
        
        # Check that migration happened
        self.assertEqual(env.migration_count, 1, "Should have one migration")
        
        # Migration overhead should be longer than regular restart overhead
        # Find the tick where migration happened (around tick 5-6)
        # and verify overhead was applied
        zero_progress_after_migration = 0
        migration_started = False
        
        for i in range(5, 10):
            if strategy.task_done_time[i] == 0:
                migration_started = True
                zero_progress_after_migration += 1
        
        self.assertTrue(migration_started, "Should have migration overhead")
        self.assertGreaterEqual(zero_progress_after_migration, 3,
                               "Migration should cause multiple ticks of overhead")


if __name__ == '__main__':
    unittest.main()