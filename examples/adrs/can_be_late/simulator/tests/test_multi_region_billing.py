"""Tests for multi-region billing system."""

import json
import os
import tempfile
import unittest
from typing import Dict, Optional, Generator

from sky_spot.env import MultiTraceEnv
from sky_spot.strategies.strategy import MultiRegionStrategy
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


class MockArgs:
    """Mock arguments for strategy initialization."""
    def __init__(self):
        self.deadline_hours = 2.0  # Increased to prevent SAFETY NET triggering
        self.restart_overhead_hours = [0.05]  # 3 minutes
        self.inter_task_overhead = [0.0]


class TestMultiRegionBilling(unittest.TestCase):
    """Test multi-region billing scenarios."""
    
    def setUp(self):
        """Create mock trace files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.gap_seconds = 60  # 1 minute per tick
        
        # Price setup based on sky_spot/utils.py
        # v100_1 base price is 3.06, spot price will be calculated as base_price / COST_K
        # where COST_K = 3.06 / 0.9731 ≈ 3.145
        self.base_price = 3.06  # ON_DEMAND price for v100_1
        self.spot_price = self.base_price / 3.145  # Approximately 0.973
        self.ondemand_price = self.base_price
        
        # Create mock traces with known availability patterns
        # Region 0: Always available
        self.create_trace_file(0, [0] * 100)  # 0 = available
        
        # Region 1: Available from tick 0-49, unavailable 50-99
        self.create_trace_file(1, [0] * 50 + [1] * 50)  # 1 = unavailable
        
        # Region 2: Alternating availability
        self.create_trace_file(2, [0, 1] * 50)
        
        self.trace_files = [
            os.path.join(self.temp_dir, f"region_{i}_v100_1.json")
            for i in range(3)
        ]
        
    def create_trace_file(self, region_id: int, availability: list):
        """Create a mock trace file."""
        trace_data = {
            "metadata": {
                "region": f"test-region-{region_id}",
                "instance_type": "test.large",
                "start_time": "2024-01-01T00:00:00Z",
                "gap_seconds": self.gap_seconds,
                "device": "v100_1"
            },
            "data": availability
            # Don't include prices - let TraceEnv calculate spot price from base price
        }
        
        filepath = os.path.join(self.temp_dir, f"region_{region_id}_v100_1.json")
        with open(filepath, 'w') as f:
            json.dump(trace_data, f)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_single_region_single_tick(self):
        """Test billing for single region, single tick."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        # Define a simple strategy that launches SPOT in region 0
        class SimpleStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = SimpleStrategy(args)
        strategy.reset(env, task)
        
        # Execute one tick
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Need to observe again to finalize costs from previous tick
        env.observe()
        
        # Check costs (1 minute of SPOT = spot_price * 60 / 3600)
        # Use the actual spot price from the environment
        actual_spot_price = env.envs[0]._spot_price
        expected_cost = actual_spot_price * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # Check cost breakdown
        breakdown = env.get_cost_breakdown()
        self.assertEqual(breakdown['tick_count'], 1)
        self.assertAlmostEqual(breakdown['by_region'][0], expected_cost, places=6)
        self.assertAlmostEqual(breakdown['by_type'][ClusterType.SPOT], expected_cost, places=6)
    
    def test_multi_region_parallel(self):
        """Test billing for multiple regions running in parallel."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ParallelStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch SPOT in region 0
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
                
                # Launch ON_DEMAND in region 1
                result = yield TryLaunch(region=1, cluster_type=ClusterType.ON_DEMAND)
                assert result.success
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = ParallelStrategy(args)
        strategy.reset(env, task)
        
        # Execute one tick
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Need to observe again to finalize costs
        env.observe()
        
        # Check costs (SPOT + ON_DEMAND for 1 minute)
        actual_spot_price = env.envs[0]._spot_price
        actual_ondemand_price = env.envs[1]._base_price
        expected_cost = (actual_spot_price + actual_ondemand_price) * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=5)  # Reduce precision
        
        # Check breakdown
        breakdown = env.get_cost_breakdown()
        self.assertAlmostEqual(
            breakdown['by_region'][0], 
            actual_spot_price * self.gap_seconds / 3600, 
            places=5  # Reduce precision requirement
        )
        self.assertAlmostEqual(
            breakdown['by_region'][1], 
            actual_ondemand_price * self.gap_seconds / 3600, 
            places=5  # Reduce precision requirement
        )
    
    def test_terminate_still_charged(self):
        """Test that terminating an instance at tick start doesn't charge for that tick."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class TerminateStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.first_tick = True
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                if self.first_tick:
                    # First tick: launch SPOT
                    self.first_tick = False
                    result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                    assert result.success
                else:
                    # Second tick: terminate (should not be charged for this tick)
                    yield Terminate(region=0)
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = TerminateStrategy(args)
        strategy.reset(env, task)
        
        # Tick 1: Launch
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Tick 2: Terminate
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Final observe to finalize costs
        env.observe()
        
        # Should only be charged for 1 tick (terminated at start of tick 2)
        actual_spot_price = env.envs[0]._spot_price
        expected_cost = 1 * actual_spot_price * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # Verify no active instances after termination
        self.assertEqual(len(env.get_active_instances()), 0)
    
    def test_failed_launch_no_charge(self):
        """Test that failed launches don't incur charges."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class FailedLaunchStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Try to launch in region 1 at tick 50 (unavailable)
                result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                assert not result.success  # Should fail
                
                # Fall back to ON_DEMAND
                result = yield TryLaunch(region=1, cluster_type=ClusterType.ON_DEMAND)
                assert result.success
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = FailedLaunchStrategy(args)
        strategy.reset(env, task)
        
        # Skip to tick 50 where region 1 SPOT is unavailable
        # Need to properly advance ticks through observe/step cycle
        for i in range(50):
            env.observe()
            env.tick += 1
        
        env.observe()
        env.execute_multi_strategy(strategy)
        env.tick += 1
        
        # Final observe to finalize costs
        env.observe()
        
        # Should only be charged for ON_DEMAND, not failed SPOT
        actual_ondemand_price = env.envs[1]._base_price
        expected_cost = actual_ondemand_price * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
    
    def test_preemption_handling(self):
        """Test that preempted instances stop charging."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class SimpleSpotStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                active = self.env.get_active_instances()
                if not active:
                    # Launch SPOT in region 1
                    result = yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                    if result.success:
                        return
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = SimpleSpotStrategy(args)
        strategy.reset(env, task)
        
        # Run for several ticks
        for i in range(55):  # Will be preempted at tick 50
            env.observe()
            
            # Strategy will try to launch if no active instances
            env.execute_multi_strategy(strategy)
            
            env.tick += 1
        
        # Final observe to get final costs
        env.observe()
        
        # Should only be charged for ticks 0-49 (50 ticks total)
        actual_spot_price = env.envs[1]._spot_price
        expected_cost = 50 * actual_spot_price * self.gap_seconds / 3600
        self.assertAlmostEqual(env.accumulated_cost, expected_cost, places=6)
        
        # Verify no active instances after preemption
        self.assertEqual(len(env.get_active_instances()), 0)
        
    def test_complex_scenario(self):
        """Test a complex scenario with multiple operations."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class ComplexStrategy(MultiRegionStrategy):
            def __init__(self, args):
                super().__init__(args)
                self.tick_count = 0
                
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                active = self.env.get_active_instances()
                
                if self.tick_count == 0:
                    # Start with SPOT in region 0
                    yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                elif self.tick_count == 2:
                    # Add redundancy with region 1
                    yield TryLaunch(region=1, cluster_type=ClusterType.SPOT)
                elif self.tick_count == 5:
                    # Terminate region 0, switch to ON_DEMAND in region 2
                    yield Terminate(region=0)
                    yield TryLaunch(region=2, cluster_type=ClusterType.ON_DEMAND)
                
                self.tick_count += 1
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)  # 1 hour task
        strategy = ComplexStrategy(args)
        strategy.reset(env, task)
        
        # Run for 10 ticks
        costs_per_tick = []
        for i in range(10):
            env.observe()
            env.execute_multi_strategy(strategy)
            env.tick += 1
            
            # Calculate cost for this tick after incrementing tick
            # (costs are finalized in the next observe)
        
        # Final observe to finalize all costs
        env.observe()
        
        # Now calculate costs per tick from the history
        cost_history = env.cost_history
        for i in range(len(cost_history)):
            tick_costs = cost_history[i]
            tick_cost = 0
            for region, ctype in tick_costs.items():
                cost_map = env.envs[region].get_constant_cost_map()
                tick_cost += cost_map[ctype] * self.gap_seconds / 3600
            costs_per_tick.append(tick_cost)
        
        # Verify costs match expected pattern:
        # Ticks 0-1: SPOT in region 0
        # Ticks 2-4: SPOT in regions 0 and 1  
        # Tick 5: SPOT in region 1 + ON_DEMAND in region 2 (region 0 terminated, not charged)
        # Ticks 6-9: SPOT in region 1 + ON_DEMAND in region 2
        
        actual_spot_price = env.envs[0]._spot_price
        actual_ondemand_price = env.envs[2]._base_price
        spot_tick_cost = actual_spot_price * self.gap_seconds / 3600
        ondemand_tick_cost = actual_ondemand_price * self.gap_seconds / 3600
        
        expected_costs = [
            spot_tick_cost,  # 0
            spot_tick_cost,  # 1
            2 * spot_tick_cost,  # 2
            2 * spot_tick_cost,  # 3
            2 * spot_tick_cost,  # 4
            spot_tick_cost + ondemand_tick_cost,  # 5 (region 0 terminated, not charged)
            spot_tick_cost + ondemand_tick_cost,  # 6
            spot_tick_cost + ondemand_tick_cost,  # 7
            spot_tick_cost + ondemand_tick_cost,  # 8
            spot_tick_cost + ondemand_tick_cost,  # 9
        ]
        
        for i, (actual, expected) in enumerate(zip(costs_per_tick, expected_costs)):
            self.assertAlmostEqual(
                actual, expected, places=6,
                msg=f"Cost mismatch at tick {i}: expected {expected}, got {actual}"
            )
        
        # Verify final state
        final_active = env.get_active_instances()
        self.assertEqual(final_active[1], ClusterType.SPOT)
        self.assertEqual(final_active[2], ClusterType.ON_DEMAND)
        self.assertNotIn(0, final_active)
    
    def test_same_tick_launch_terminate_error(self):
        """Test that launching and terminating in the same tick raises an error."""
        env = MultiTraceEnv(self.trace_files, env_start_hours=0)
        
        class SameTickStrategy(MultiRegionStrategy):
            def _step_multi(self) -> Generator[Action, Optional[LaunchResult], None]:
                # Launch SPOT instance
                result = yield TryLaunch(region=0, cluster_type=ClusterType.SPOT)
                assert result.success
                
                # Try to terminate in the same tick (should raise error)
                yield Terminate(region=0)
        
        args = MockArgs()
        task = MockTask(duration_seconds=3600)
        strategy = SameTickStrategy(args)
        strategy.reset(env, task)
        
        # Should raise ValueError
        env.observe()
        with self.assertRaises(ValueError) as cm:
            env.execute_multi_strategy(strategy)
        
        self.assertIn("Cannot terminate SPOT instance in region 0 in the same tick it was launched", str(cm.exception))
        self.assertIn("Minimum billing unit is one tick", str(cm.exception))


if __name__ == '__main__':
    unittest.main() 