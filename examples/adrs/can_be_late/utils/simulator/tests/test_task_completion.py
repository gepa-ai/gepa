"""Tests for task completion boundary conditions."""

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
from sky_spot import simulate


class SimpleTask(task_lib.Task):
    """Simple task for testing."""
    
    def __init__(self, duration_seconds: float):
        self.duration_seconds = duration_seconds
        self._progress_source = []
        self.checkpoint_size_gb = 1.0
    
    def set_progress_source(self, progress_source):
        self._progress_source = progress_source
    
    def get_total_duration_seconds(self) -> float:
        return self.duration_seconds
    
    def get_total_duration_hours(self) -> float:
        return self.duration_seconds / 3600.0
    
    def get_remaining_duration_seconds(self) -> float:
        return max(0.0, self.duration_seconds - sum(self._progress_source))
    
    def get_current_progress_seconds(self) -> float:
        return sum(self._progress_source)
    
    def get_info(self) -> dict:
        progress = sum(self._progress_source)
        return {
            'progress': progress,
            'remaining': self.duration_seconds - progress,
            'total': self.duration_seconds
        }
    
    @property
    def is_done(self) -> bool:
        return self.get_remaining_duration_seconds() <= 1e-8
    
    def get_config(self) -> dict:
        return {'duration_seconds': self.duration_seconds}
    
    def reset(self):
        """Reset the task state."""
        self._progress_source = []
    
    def __str__(self) -> str:
        """String representation of the task."""
        return f"SimpleTask(duration={self.duration_seconds}s)"


class SimpleMultiRegionStrategy(MultiRegionStrategy):
    """Simple strategy that launches SPOT in region 0 and never terminates."""
    NAME = 'simple_multi_test'
    
    def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
        # Check if task is done
        if self.task_done:
            # Terminate any active instances
            active = self.env.get_active_instances()
            for region in active:
                yield Terminate(region=region)
            return
        
        # Check if we have an active instance
        active = self.env.get_active_instances()
        if not active:
            # Launch SPOT in region 0
            result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
            assert result is not None
            assert result.success, "SPOT should be available in test"


class TestTaskCompletion(unittest.TestCase):
    """Test task completion boundary conditions."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 60  # 1 minute per tick
        
        # Create a trace file with always available spot
        trace_data = {
            "metadata": {
                "region": "test-region-0",
                "instance_type": "test.large",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            # Create enough ticks for the test (2000 ticks = 2000 minutes)
            "data": [0] * 2000  # 0 = available
        }
        
        self.trace_file = os.path.join(self.temp_dir, "region_0_v100_1.json")
        with open(self.trace_file, 'w') as f:
            json.dump(trace_data, f)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_multi_region_task_completion_boundary(self):
        """Test that multi-region simulation stops when task completes."""
        # Create environment
        env = MultiTraceEnv([self.trace_file], env_start_hours=0)
        
        # Create a task that will complete in exactly 10 ticks
        # Each tick is 60 seconds, so 10 ticks = 600 seconds
        task = SimpleTask(duration_seconds=600)
        
        # Create strategy
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--deadline-hours', default=1.0, type=float)
        parser.add_argument('--restart-overhead-hours', nargs='+', default=[0.0], type=float)
        parser.add_argument('--keep-on-demand', default=None)
        args = parser.parse_args([])
        
        strategy = SimpleMultiRegionStrategy(args)
        strategy.reset(env, task)
        
        # Track ticks
        tick_count = 0
        progress_history = []
        
        # Run simulation manually to track behavior
        while not strategy.task_done and tick_count < 20:  # Safety limit
            env.observe()
            env.update_strategy_progress(strategy)
            
            # Record progress
            progress = task.get_current_progress_seconds()
            progress_history.append(progress)
            
            # Check if task became done after progress update
            if strategy.task_done:
                break
            
            env.execute_multi_strategy(strategy)
            env.tick += 1
            tick_count += 1
        
        # Verify task completed
        self.assertTrue(strategy.task_done, "Task should be done")
        self.assertEqual(task.get_current_progress_seconds(), 600, 
                         "Task should have completed exactly 600 seconds of work")
        
        # Verify simulation stopped at the right time
        # With cold start overhead of 0, work should start immediately
        # So it should complete after exactly 10 ticks
        self.assertLessEqual(tick_count, 11, 
                            f"Simulation should stop soon after task completion, but ran for {tick_count} ticks")
        
        # Verify no work was done after task completion
        self.assertLessEqual(sum(strategy.task_done_time), 600,
                            "Should not accumulate more progress than task duration")
    
    def test_task_completion_with_overhead(self):
        """Test task completion with restart overhead."""
        # Create environment
        env = MultiTraceEnv([self.trace_file], env_start_hours=0)
        
        # Create a task that needs 600 seconds of work
        task = SimpleTask(duration_seconds=600)
        
        # Create strategy with restart overhead
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--deadline-hours', default=1.0, type=float)
        # 0.05 hours = 3 minutes = 180 seconds = 3 ticks
        parser.add_argument('--restart-overhead-hours', nargs='+', default=[0.05], type=float)
        parser.add_argument('--keep-on-demand', default=None)
        args = parser.parse_args([])
        
        strategy = SimpleMultiRegionStrategy(args)
        strategy.reset(env, task)
        
        # Run simulation
        tick_count = 0
        while not strategy.task_done and tick_count < 30:  # Safety limit
            env.observe()
            env.update_strategy_progress(strategy)
            
            if strategy.task_done:
                break
            
            env.execute_multi_strategy(strategy)
            env.tick += 1
            tick_count += 1
        
        # Verify task completed
        self.assertTrue(strategy.task_done, "Task should be done")
        
        # With 3 ticks of overhead, work starts at tick 4
        # 10 ticks of work needed, so should complete around tick 13
        self.assertLessEqual(tick_count, 15,
                            f"Simulation should stop soon after task completion, but ran for {tick_count} ticks")
    
    def test_task_near_trace_boundary(self):
        """Test task completion near the end of trace data."""
        # Create a short trace file
        trace_data = {
            "metadata": {
                "region": "test-region-0",
                "instance_type": "test.large",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            # Only 100 ticks of data
            "data": [0] * 100
        }
        
        short_trace_file = os.path.join(self.temp_dir, "short_trace_v100_1.json")
        with open(short_trace_file, 'w') as f:
            json.dump(trace_data, f)
        
        # Create environment
        env = MultiTraceEnv([short_trace_file], env_start_hours=0)
        
        # Create a task that will complete at tick 95 (before trace ends)
        task = SimpleTask(duration_seconds=95 * self.gap_seconds)
        
        # Create strategy
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--deadline-hours', default=2.0, type=float)
        parser.add_argument('--restart-overhead-hours', nargs='+', default=[0.0], type=float)
        parser.add_argument('--keep-on-demand', default=None)
        args = parser.parse_args([])
        
        strategy = SimpleMultiRegionStrategy(args)
        strategy.reset(env, task)
        
        # Run simulation
        tick_count = 0
        error_occurred = False
        
        try:
            while not strategy.task_done and tick_count < 110:  # Past trace boundary
                env.observe()
                env.update_strategy_progress(strategy)
                
                if strategy.task_done:
                    break
                
                env.execute_multi_strategy(strategy)
                env.tick += 1
                tick_count += 1
        except ValueError as e:
            if "exceeded trace data bounds" in str(e):
                error_occurred = True
        
        # Verify task completed without hitting trace boundary
        self.assertTrue(strategy.task_done, "Task should be done")
        self.assertFalse(error_occurred, "Should not hit trace boundary error")
        self.assertLess(tick_count, 98, "Should complete before trace boundary")


    def test_single_region_task_completion_boundary(self):
        """Test that single-region simulation stops when task completes."""
        # Create environment
        from sky_spot.env import TraceEnv
        env = TraceEnv(self.trace_file, env_start_hours=0)
        
        # Create a task that will complete in exactly 10 ticks
        task = SimpleTask(duration_seconds=600)
        
        # Create a simple single-region strategy
        from sky_spot.strategies.strategy import Strategy
        from sky_spot.utils import ClusterType
        
        class SimpleSpotStrategy(Strategy):
            NAME = 'simple_spot_test'
            
            def _step(self, last_cluster_type: ClusterType, has_spot: bool) -> ClusterType:
                # Always request SPOT if available
                if has_spot:
                    return ClusterType.SPOT
                else:
                    return ClusterType.ON_DEMAND
            
            @classmethod
            def _from_args(cls, parser):
                args = parser.parse_args([])
                return cls(args)
        
        # Create strategy
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--deadline-hours', default=1.0, type=float)
        parser.add_argument('--restart-overhead-hours', nargs='+', default=[0.0], type=float)
        parser.add_argument('--keep-on-demand', default=None)
        args = parser.parse_args([])
        
        strategy = SimpleSpotStrategy(args)
        env.reset()
        strategy.reset(env, task)
        
        # Run simulation
        tick_count = 0
        while not strategy.task_done and tick_count < 20:  # Safety limit
            request_type = strategy.step()
            env.step(request_type)
            
            # Check if task became done
            if strategy.task_done:
                break
                
            tick_count += 1
        
        # Verify task completed
        self.assertTrue(strategy.task_done, "Task should be done")
        self.assertEqual(task.get_current_progress_seconds(), 600, 
                         "Task should have completed exactly 600 seconds of work")
        
        # Should stop at or before tick 10 (0-indexed)
        self.assertLessEqual(tick_count, 10,
                            f"Simulation should stop at task completion, but ran for {tick_count} ticks")


if __name__ == '__main__':
    unittest.main()