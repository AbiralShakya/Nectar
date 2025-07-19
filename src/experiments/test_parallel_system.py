#!/usr/bin/env python3
"""
Test Suite for Parallel Energy-Aware MoE System

This script provides comprehensive tests for the parallel MoE system to ensure
all components work correctly before running full experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.parallel_moe_system import (
    ParallelMoEConfig, ParallelMoELayer, create_parallel_moe_system,
    GlobalLoadBalancer, EnergyAwareScheduler, ParallelExpertPool
)
from src.moe_models import MoEConfig
from src.monitor import GpuSystemMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelMoETestSuite:
    """Comprehensive test suite for parallel MoE system"""
    
    def __init__(self):
        self.test_results = {}
        self.device_ids = [0] if torch.cuda.is_available() else [0]
        
        # Create test configuration
        self.test_config = self._create_test_config()
        
        logger.info(f"Initialized test suite with device_ids: {self.device_ids}")
    
    def _create_test_config(self) -> ParallelMoEConfig:
        """Create test configuration"""
        moe_config = MoEConfig(
            d_model=256,  # Smaller for testing
            num_experts=4,  # Fewer experts for testing
            top_k=2,
            expert_type="swiglu",
            batch_size=8  # Smaller batch for testing
        )
        
        return ParallelMoEConfig(
            moe_config=moe_config,
            world_size=1,
            energy_budget_watts=200.0,
            thermal_threshold_celsius=70.0,
            joules_per_token_target=0.001,
            rerouting_enabled=True,
            ttt_enabled=True,
            async_expert_execution=True,
            mixed_precision=False  # Disable for testing stability
        )
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        logger.info("Starting comprehensive test suite...")
        
        tests = [
            ("test_basic_moe_creation", self.test_basic_moe_creation),
            ("test_forward_pass", self.test_forward_pass),
            ("test_energy_aware_routing", self.test_energy_aware_routing),
            ("test_dynamic_rerouting", self.test_dynamic_rerouting),
            ("test_ttt_adaptation", self.test_ttt_adaptation),
            ("test_load_balancer", self.test_load_balancer),
            ("test_energy_scheduler", self.test_energy_scheduler),
            ("test_hardware_monitoring", self.test_hardware_monitoring),
            ("test_async_execution", self.test_async_execution),
            ("test_performance_tracking", self.test_performance_tracking)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                start_time = time.time()
                result = await test_func()
                end_time = time.time()
                
                self.test_results[test_name] = {
                    'passed': result,
                    'duration': end_time - start_time,
                    'error': None
                }
                
                if result:
                    logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
                self.test_results[test_name] = {
                    'passed': False,
                    'duration': 0,
                    'error': str(e)
                }
                all_passed = False
        
        # Print summary
        self._print_test_summary()
        
        return all_passed
    
    async def test_basic_moe_creation(self) -> bool:
        """Test basic MoE layer creation"""
        try:
            moe_layer = create_parallel_moe_system(self.test_config)
            
            # Check if layer was created successfully
            assert moe_layer is not None
            assert hasattr(moe_layer, 'expert_pool')
            assert hasattr(moe_layer, 'load_balancer')
            assert hasattr(moe_layer, 'gate')
            
            # Check configuration
            assert moe_layer.config.moe_config.num_experts == 4
            assert moe_layer.config.moe_config.d_model == 256
            
            return True
            
        except Exception as e:
            logger.error(f"Basic MoE creation failed: {e}")
            return False
    
    async def test_forward_pass(self) -> bool:
        """Test basic forward pass"""
        try:
            moe_layer = create_parallel_moe_system(self.test_config)
            
            # Create test input
            batch_size = 4
            seq_len = 32
            d_model = 256
            
            x = torch.randn(batch_size, seq_len, d_model)
            if torch.cuda.is_available():
                x = x.cuda()
            
            # Forward pass
            output = await moe_layer(x)
            
            # Check output shape
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            return True
            
        except Exception as e:
            logger.error(f"Forward pass test failed: {e}")
            return False
    
    async def test_energy_aware_routing(self) -> bool:
        """Test energy-aware routing functionality"""
        try:
            # Create config with energy awareness enabled
            config = self.test_config
            config.power_efficiency_weight = 0.5
            
            moe_layer = create_parallel_moe_system(config)
            
            # Test multiple forward passes to see routing adaptation
            x = torch.randn(4, 32, 256)
            if torch.cuda.is_available():
                x = x.cuda()
            
            outputs = []
            for _ in range(5):
                output = await moe_layer(x)
                outputs.append(output)
            
            # Check that outputs are reasonable
            for output in outputs:
                assert not torch.isnan(output).any()
                assert output.shape == x.shape
            
            # Check that performance stats are being tracked
            stats = moe_layer.get_performance_stats()
            assert 'batch_count' in stats
            assert stats['batch_count'] >= 5
            
            return True
            
        except Exception as e:
            logger.error(f"Energy-aware routing test failed: {e}")
            return False
    
    async def test_dynamic_rerouting(self) -> bool:
        """Test dynamic expert rerouting"""
        try:
            config = self.test_config
            config.rerouting_enabled = True
            config.rerouting_history_length = 10
            
            moe_layer = create_parallel_moe_system(config)
            
            # Run multiple batches to build up history
            x = torch.randn(4, 32, 256)
            if torch.cuda.is_available():
                x = x.cuda()
            
            for i in range(15):  # More than history length
                output = await moe_layer(x)
                assert not torch.isnan(output).any()
            
            # Check that batch distribution tracker has history
            tracker = moe_layer.load_balancer.batch_tracker
            assert len(tracker.distribution_history) > 0
            
            # Test rerouting bias computation
            current_dist = torch.ones(4) / 4  # Uniform distribution
            hardware_metrics = {'temperature': 60.0, 'power_watt': 150.0}
            
            biases, metadata = tracker.compute_rerouting_biases(
                current_dist, hardware_metrics
            )
            
            assert biases.shape == (4,)  # One bias per expert
            assert 'current_imbalance' in metadata
            
            return True
            
        except Exception as e:
            logger.error(f"Dynamic rerouting test failed: {e}")
            return False
    
    async def test_ttt_adaptation(self) -> bool:
        """Test TTT adaptation functionality"""
        try:
            config = self.test_config
            config.ttt_enabled = True
            config.ttt_update_frequency = 5  # Update every 5 batches
            
            moe_layer = create_parallel_moe_system(config)
            
            # Check TTT adapter exists
            assert hasattr(moe_layer, 'ttt_adapter')
            assert moe_layer.ttt_adapter is not None
            
            # Run enough batches to trigger TTT update
            x = torch.randn(4, 32, 256)
            if torch.cuda.is_available():
                x = x.cuda()
            
            for i in range(10):
                output = await moe_layer(x)
                assert not torch.isnan(output).any()
            
            # Check that batch count increased
            assert moe_layer.batch_count >= 10
            
            return True
            
        except Exception as e:
            logger.error(f"TTT adaptation test failed: {e}")
            return False
    
    async def test_load_balancer(self) -> bool:
        """Test global load balancer"""
        try:
            load_balancer = GlobalLoadBalancer(self.test_config, self.device_ids)
            
            # Test routing computation
            batch_size = 8
            num_experts = 4
            routing_logits = torch.randn(batch_size, num_experts)
            
            # Mock hardware states
            hardware_states = {
                0: {
                    'temperature': 60.0,
                    'power_watt': 150.0,
                    'gpu_utilization_percent': 70.0,
                    'memory_utilization_percent': 60.0
                }
            }
            
            current_load = {i: np.random.uniform(0.1, 0.9) for i in range(num_experts)}
            
            # Compute optimal routing
            topk_indices, topk_probs, metadata = load_balancer.compute_optimal_routing(
                routing_logits, hardware_states, current_load
            )
            
            # Check outputs
            assert topk_indices.shape == (batch_size, 2)  # top_k = 2
            assert topk_probs.shape == (batch_size, 2)
            assert 'energy_biases' in metadata
            assert 'load_biases' in metadata
            
            # Check probability normalization
            prob_sums = topk_probs.sum(dim=1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
            
            return True
            
        except Exception as e:
            logger.error(f"Load balancer test failed: {e}")
            return False
    
    async def test_energy_scheduler(self) -> bool:
        """Test energy-aware scheduler"""
        try:
            scheduler = EnergyAwareScheduler(self.test_config, self.device_ids)
            
            # Mock expert assignments
            expert_assignments = {
                0: torch.randn(10, 256),
                1: torch.randn(8, 256),
                2: torch.randn(12, 256),
                3: torch.randn(6, 256)
            }
            
            # Mock hardware states
            hardware_states = {
                0: {
                    'temperature': 65.0,
                    'power_watt': 180.0,
                    'gpu_utilization_percent': 75.0
                }
            }
            
            # Create execution plan
            execution_plan = scheduler.create_execution_plan(
                expert_assignments, hardware_states
            )
            
            # Check execution plan
            assert isinstance(execution_plan, dict)
            assert 0 in execution_plan  # Device 0 should be in plan
            
            # Check that all experts are assigned
            total_experts_assigned = sum(len(tasks) for tasks in execution_plan.values())
            assert total_experts_assigned == len(expert_assignments)
            
            return True
            
        except Exception as e:
            logger.error(f"Energy scheduler test failed: {e}")
            return False
    
    async def test_hardware_monitoring(self) -> bool:
        """Test hardware monitoring integration"""
        try:
            # Test GPU monitor creation
            monitor = GpuSystemMonitor(0)
            
            # Get current stats
            stats = monitor.get_current_stats()
            
            # Check required fields
            required_fields = ['temperature', 'power_watt', 'thermal_state']
            for field in required_fields:
                assert field in stats
            
            # Check reasonable values
            assert 0 <= stats['temperature'] <= 150  # Reasonable temp range
            assert 0 <= stats['power_watt'] <= 1000   # Reasonable power range
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware monitoring test failed: {e}")
            return False
    
    async def test_async_execution(self) -> bool:
        """Test async expert execution"""
        try:
            config = self.test_config
            config.async_expert_execution = True
            
            expert_pool = ParallelExpertPool(config, self.device_ids)
            
            # Mock expert assignments
            expert_assignments = {
                0: torch.randn(5, 256),
                1: torch.randn(3, 256)
            }
            
            input_tokens = torch.randn(10, 256)
            
            # Test async execution
            results = await expert_pool.forward_experts_async(
                expert_assignments, input_tokens
            )
            
            # Check results
            assert isinstance(results, dict)
            assert len(results) <= len(expert_assignments)
            
            # Check output shapes
            for expert_id, output in results.items():
                assert output.shape[1] == 256  # d_model dimension
            
            return True
            
        except Exception as e:
            logger.error(f"Async execution test failed: {e}")
            return False
    
    async def test_performance_tracking(self) -> bool:
        """Test performance tracking and metrics"""
        try:
            moe_layer = create_parallel_moe_system(self.test_config)
            
            # Run several forward passes
            x = torch.randn(4, 32, 256)
            if torch.cuda.is_available():
                x = x.cuda()
            
            for _ in range(5):
                await moe_layer(x)
            
            # Get performance stats
            stats = moe_layer.get_performance_stats()
            
            # Check required metrics
            required_metrics = ['batch_count', 'num_devices']
            for metric in required_metrics:
                assert metric in stats
            
            # Check reasonable values
            assert stats['batch_count'] >= 5
            assert stats['num_devices'] >= 1
            
            # Check energy consumption log
            assert len(moe_layer.energy_consumption_log) > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Performance tracking test failed: {e}")
            return False
    
    def _print_test_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        print()
        
        if failed_tests > 0:
            print("FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if not result['passed']:
                    error_msg = result['error'] if result['error'] else "Unknown error"
                    print(f"  ❌ {test_name}: {error_msg}")
            print()
        
        print("DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = result['duration']
            print(f"  {status} {test_name:<30} ({duration:.2f}s)")
        
        print("="*60)

async def main():
    """Main test runner"""
    print("Parallel Energy-Aware MoE System Test Suite")
    print("=" * 50)
    
    # Check system requirements
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Run tests
    test_suite = ParallelMoETestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The system is ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)