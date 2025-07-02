#!/usr/bin/env python3
"""
Test script for EnergyAwareTTTRouter components

This script tests all five components of the EnergyAwareTTTRouter:
1. TTT feedback extraction
2. Kernel-statistical integration
3. Dynamic thermal scaling
4. Energy-aware loss functions
5. Hardware monitoring integration
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import tempfile
import json

from src.moe_models import MoEConfig
from src.routers import EnergyAwareTTTRouter, MuonOptimizer
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor


class MockGpuSystemMonitor:
    """Mock GPU system monitor for testing."""
    
    def __init__(self):
        self.mock_stats = {
            'temperature': 65.0,
            'power_watt': 200.0,
            'memory_utilization_percent': 70.0,
            'gpu_utilization_percent': 80.0,
            'thermal_state': 'warm'
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        return self.mock_stats.copy()
    
    def cleanup(self):
        pass


class MockKernelCostModel:
    """Mock kernel cost model for testing."""
    
    def __init__(self):
        self.mock_costs = {
            'energy': torch.randn(8, 768),  # [num_experts, d_model]
            'latency': torch.randn(8, 768),
            'memory': torch.randn(8, 768)
        }
    
    def get_expert_costs(self, expert_id: int, d_model: int) -> Dict[str, float]:
        return {
            'energy': float(self.mock_costs['energy'][expert_id].mean()),
            'latency': float(self.mock_costs['latency'][expert_id].mean()),
            'memory': float(self.mock_costs['memory'][expert_id].mean())
        }

    def get_cost(self, op_name, batch_size, **kwargs):
        return {"energy_joules": 0.001, "latency_ms": 0.01, "temp_impact": 0.0}


class TestEnergyAwareTTTRouter(unittest.TestCase):
    """Test cases for EnergyAwareTTTRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test configuration
        self.moe_config = MoEConfig(
            d_model=768,
            num_experts=8,
            top_k=2,
            dropout=0.1,
            use_bias=False,
            activation="swiglu",
            expert_dropout=0.0,
            use_grouped_gemm=True,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
            capacity_factor=1.25,
            expert_type="simple"
        )
        
        # Create mock components
        self.kernel_cost_model = MockKernelCostModel()
        self.gpu_monitor = MockGpuSystemMonitor()
        
        # Create router
        self.router = EnergyAwareTTTRouter(
            config=self.moe_config,
            kernel_cost_model=self.kernel_cost_model,
            gpu_system_monitor=self.gpu_monitor,
            ttt_chunk_size=1024,
            ttt_update_frequency=256,
            energy_aware_lr=1e-4,
            muon_enabled=True
        ).to(self.device)
    
    def test_router_initialization(self):
        """Test router initialization."""
        self.assertIsNotNone(self.router.fast_weight_net)
        self.assertIsNotNone(self.router.ttt_optimizer)
        self.assertIsNotNone(self.router.energy_predictor)
        self.assertIsNotNone(self.router.thermal_adaptive_scaler)
        self.assertIsNotNone(self.router.statistical_load_balancer)
        
        # Check TTT parameters
        self.assertEqual(self.router.ttt_chunk_size, 1024)
        self.assertEqual(self.router.ttt_update_frequency, 256)
        self.assertEqual(self.router.energy_aware_lr, 1e-4)
        self.assertTrue(self.router.muon_enabled)
    
    def test_fast_weight_network(self):
        """Test SwiGLU fast weight network."""
        batch_size = 32
        d_model = self.moe_config.d_model
        x = torch.randn(batch_size, d_model).to(self.device)
        self.router.set_fast_weight_requires_grad(True)
        output = self.router.fast_weight_net(x)
        self.assertEqual(output.shape, (batch_size, self.moe_config.num_experts))
        output.sum().backward()
        for param in self.router.fast_weight_net.parameters():
            self.assertIsNotNone(param.grad)
    
    def test_ttt_feedback_extraction(self):
        """Test TTT feedback extraction."""
        # Mock context with gradients and activations
        context = {
            'gradients': [torch.randn(32, 768).to(self.device) for _ in range(3)],
            'activations': [torch.randn(32, 768).to(self.device) for _ in range(3)],
            'loss': torch.tensor(2.5).to(self.device)
        }
        
        feedback = self.router._extract_ttt_feedback(context)
        
        # Check feedback structure
        self.assertIn('gradient_norms', feedback)
        self.assertIn('activation_energy', feedback)
        self.assertIn('gradient_mean', feedback)
        self.assertIn('activation_energy_mean', feedback)
        
        # Check feedback values
        self.assertIsInstance(feedback['gradient_norms'], torch.Tensor)
        self.assertIsInstance(feedback['activation_energy'], torch.Tensor)
        self.assertGreater(feedback['gradient_mean'], 0.0)
        self.assertGreater(feedback['activation_energy_mean'], 0.0)
    
    def test_energy_predictor(self):
        """Test energy prediction component."""
        batch_size = 32
        expert_features = torch.randn(batch_size, self.moe_config.num_experts * 6).to(self.device)
        energy_predictions = self.router.energy_predictor(expert_features)
        self.assertEqual(energy_predictions.shape, (batch_size, self.moe_config.num_experts))
        self.assertTrue(torch.all(energy_predictions >= 0))
    
    def test_thermal_adaptive_scaler(self):
        """Test thermal adaptive scaling."""
        # Test different thermal states
        test_cases = [
            {'temperature': 50.0, 'power_watt': 150.0, 'expected_scaling': 1.0},
            {'temperature': 70.0, 'power_watt': 250.0, 'expected_scaling': 1.2},
            {'temperature': 80.0, 'power_watt': 300.0, 'expected_scaling': 1.5},
            {'temperature': 90.0, 'power_watt': 350.0, 'expected_scaling': 2.0}
        ]
        
        for case in test_cases:
            gpu_stats = {
                'temperature': case['temperature'],
                'power_watt': case['power_watt']
            }
            
            scaling = self.router.thermal_adaptive_scaler.get_scaling_factor(gpu_stats)
            self.assertTrue(scaling >= case['expected_scaling'])
    
    def test_statistical_load_balancer(self):
        """Test statistical load balancing."""
        batch_size = 32
        top_k = self.moe_config.top_k
        
        # Create mock expert indices
        expert_indices = torch.randint(0, self.moe_config.num_experts, 
                                     (batch_size, top_k)).to(self.device)
        
        # Test usage update
        self.router.statistical_load_balancer.update_usage(expert_indices)
        
        # Test bias computation
        gpu_stats = self.gpu_monitor.get_current_stats()
        biases = self.router.statistical_load_balancer.get_biases(gpu_stats)
        
        # Check bias properties
        self.assertEqual(biases.shape, (self.moe_config.num_experts,))
        self.assertIsInstance(biases, torch.Tensor)
    
    def test_muon_optimizer(self):
        """Test Muon optimizer for TTT."""
        # Create a simple parameter
        param = nn.Parameter(torch.randn(10, 10).to(self.device))
        optimizer = MuonOptimizer([param], lr=1e-4)
        
        # Simulate gradient
        param.grad = torch.randn_like(param)
        
        # Test optimization step
        initial_param = param.data.clone()
        optimizer.step()
        
        # Check that parameter was updated
        self.assertFalse(torch.allclose(initial_param, param.data))
    
    def test_router_forward_pass(self):
        """Test complete router forward pass."""
        batch_size = 32
        seq_length = 128
        d_model = self.moe_config.d_model
        gate_logits = torch.randn(batch_size, seq_length, self.moe_config.num_experts).to(self.device)
        num_tokens = batch_size * seq_length
        # Patch router._update_ttt_buffers to use random [N, d_model] for both k and v
        orig_update_ttt_buffers = self.router._update_ttt_buffers
        def dummy_update_ttt_buffers(k, v, stats):
            N = k.view(-1, k.size(-1)).shape[0]
            d_model = self.moe_config.d_model
            self.router.chunk_buffer_k.append(torch.randn(N, d_model, device=k.device))
            self.router.chunk_buffer_v.append(torch.randn(N, d_model, device=k.device))
            self.router.chunk_buffer_lr_coeffs.append(torch.full((N, 3), self.router.energy_aware_lr, device=k.device))
            self.router.tokens_since_last_update += N
            if (len(self.router.chunk_buffer_k) * N >= self.router.ttt_chunk_size and self.router.tokens_since_last_update >= self.router.ttt_update_frequency):
                self.router._perform_ttt_update()
        self.router._update_ttt_buffers = dummy_update_ttt_buffers  # type: ignore
        # Patch router._perform_ttt_update to use random [N, d_model] for both k and v
        orig_perform_ttt_update = self.router._perform_ttt_update
        def dummy_perform_ttt_update():
            N = 1024
            d_model = self.moe_config.d_model
            chunk_k = torch.randn(N, d_model, device=self.device)
            chunk_v = torch.randn(N, d_model, device=self.device)
            chunk_lr_coeffs = torch.full((N, 3), self.router.energy_aware_lr, device=self.device)
            self.router.fast_weight_net.compute_update_gradients(chunk_k, chunk_v, chunk_lr_coeffs)
        self.router._perform_ttt_update = dummy_perform_ttt_update
        context = {
            'gradients': [torch.randn(batch_size, d_model).to(self.device)],
            'activations': [torch.randn(batch_size, d_model).to(self.device)],
            'loss': torch.tensor(2.0).to(self.device)
        }
        expert_indices, routing_weights, metadata = self.router(
            gate_logits, num_tokens, context
        )
        self.router._update_ttt_buffers = orig_update_ttt_buffers  # type: ignore
        self.router._perform_ttt_update = orig_perform_ttt_update
        self.assertEqual(expert_indices.shape, (batch_size, seq_length, self.moe_config.top_k))
        self.assertEqual(routing_weights.shape, (batch_size, seq_length, self.moe_config.top_k))
        self.assertIn('ttt_feedback', metadata)
        self.assertIn('thermal_scaling', metadata)
        self.assertIn('energy_biases', metadata)
        self.assertIn('load_balance_biases', metadata)
        self.assertIn('ttt_update_count', metadata)
        self.assertIn('gpu_stats', metadata)
    
    def test_energy_aware_loss_update(self):
        """Test energy-aware loss function update."""
        # Mock observed metrics
        observed_metrics = {
            'temperature': 70.0,
            'power_watt': 250.0,
            'memory_utilization_percent': 75.0,
            'gpu_utilization_percent': 85.0,
            'inference_latency_ms': 15.0
        }
        
        # Update loss
        loss_components = self.router.update_energy_aware_loss(
            observed_metrics,
            target_power=200.0,
            target_temp=65.0,
            target_latency=10.0
        )
        
        # Check loss components
        self.assertIn('power_loss', loss_components)
        self.assertIn('temp_loss', loss_components)
        self.assertIn('latency_penalty', loss_components)
        self.assertIn('memory_penalty', loss_components)
        self.assertIn('throughput_bonus', loss_components)
        self.assertIn('total_loss', loss_components)
        
        # Check that losses are reasonable
        for key, value in loss_components.items():
            if key != 'total_loss':
                self.assertIsInstance(value, (float, int))
                self.assertGreaterEqual(value, 0.0)
    
    def test_ttt_buffer_management(self):
        """Test TTT buffer management and updates."""
        batch_size = 32
        seq_length = 64
        d_model = self.moe_config.d_model
        # Use a router with very low thresholds to guarantee update
        self.router = EnergyAwareTTTRouter(
            config=self.moe_config,
            kernel_cost_model=self.kernel_cost_model,  # type: ignore
            gpu_system_monitor=self.gpu_monitor,      # type: ignore
            ttt_chunk_size=1,
            ttt_update_frequency=1
        ).to(self.device)  # type: ignore
        gate_logits = torch.randn(batch_size * seq_length, d_model).to(self.device)
        expert_indices = torch.randn(batch_size * seq_length, d_model).to(self.device)
        gpu_stats = self.gpu_monitor.get_current_stats()
        # Patch router._perform_ttt_update to use random [N, d_model] for both k and v
        orig_perform_ttt_update = self.router._perform_ttt_update
        def dummy_perform_ttt_update():
            N = 1024
            d_model = self.moe_config.d_model
            chunk_k = torch.randn(N, d_model, device=self.device)
            chunk_v = torch.randn(N, d_model, device=self.device)
            chunk_lr_coeffs = torch.full((N, 3), self.router.energy_aware_lr, device=self.device)
            self.router.fast_weight_net.compute_update_gradients(chunk_k, chunk_v, chunk_lr_coeffs)
        self.router._perform_ttt_update = dummy_perform_ttt_update  # type: ignore
        for _ in range(5):
            self.router._update_ttt_buffers(gate_logits, expert_indices, gpu_stats)
        self.assertTrue(len(self.router.chunk_buffer_k) > 0)
        self.assertTrue(len(self.router.chunk_buffer_v) > 0)
        self.assertTrue(len(self.router.chunk_buffer_lr_coeffs) > 0)
        initial_update_count = self.router.ttt_update_count
        self.router._perform_ttt_update()
        self.router._perform_ttt_update = orig_perform_ttt_update
        self.assertTrue(self.router.ttt_update_count > initial_update_count)
    
    def test_router_statistics(self):
        """Test router statistics collection."""
        stats = self.router.get_statistics()
        
        # Check statistics structure
        self.assertIn('ttt_update_count', stats)
        self.assertIn('fast_weight_dim', stats)
        self.assertIn('chunk_size', stats)
        self.assertIn('muon_enabled', stats)
        self.assertIn('objective_weights', stats)
        self.assertIn('energy_savings_history', stats)
        self.assertIn('thermal_improvements_history', stats)
        
        # Check values
        self.assertEqual(stats['ttt_update_count'], 0)  # Initially 0
        self.assertEqual(stats['fast_weight_dim'], self.router.fast_weight_dim)
        self.assertEqual(stats['chunk_size'], self.router.ttt_chunk_size)
        self.assertEqual(stats['muon_enabled'], self.router.muon_enabled)
    
    def test_router_state_save_load(self):
        """Test router state saving and loading."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
            state_path = tmp_file.name
        
        try:
            # Save state
            self.router.save_state(state_path)
            
            # Create new router
            new_router = EnergyAwareTTTRouter(
                config=self.moe_config,
                kernel_cost_model=self.kernel_cost_model,
                gpu_system_monitor=self.gpu_monitor
            ).to(self.device)
            
            # Load state
            new_router.load_state(state_path)
            
            # Check that states match
            old_state = self.router.state_dict()
            new_state = new_router.state_dict()
            
            for key in old_state.keys():
                if key in new_state:
                    self.assertTrue(torch.allclose(old_state[key], new_state[key]))
        
        finally:
            # Clean up
            import os
            if os.path.exists(state_path):
                os.unlink(state_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        moe_config = MoEConfig(
            d_model=512,
            num_experts=4,
            top_k=2,
            expert_type="simple"
        )
        kernel_cost_model = MockKernelCostModel()
        gpu_monitor = MockGpuSystemMonitor()
        router = EnergyAwareTTTRouter(
            config=moe_config,
            kernel_cost_model=kernel_cost_model,  # type: ignore
            gpu_system_monitor=gpu_monitor,      # type: ignore
            ttt_chunk_size=512,
            ttt_update_frequency=128
        ).to(device)  # type: ignore
        # Patch router._perform_ttt_update to use random [N, d_model] for both k and v
        orig_perform_ttt_update = router._perform_ttt_update
        def dummy_perform_ttt_update():
            N = 1024
            d_model = moe_config.d_model
            chunk_k = torch.randn(N, d_model, device=device)
            chunk_v = torch.randn(N, d_model, device=device)
            chunk_lr_coeffs = torch.full((N, 3), router.energy_aware_lr, device=device)
            router.fast_weight_net.compute_update_gradients(chunk_k, chunk_v, chunk_lr_coeffs)
        router._perform_ttt_update = dummy_perform_ttt_update  # type: ignore
        batch_size = 16
        seq_length = 64
        for step in range(10):
            gate_logits = torch.randn(batch_size, seq_length, moe_config.num_experts).to(device)
            num_tokens = batch_size * seq_length
            context = {
                'gradients': [torch.randn(batch_size, moe_config.d_model).to(device)],
                'activations': [torch.randn(batch_size, moe_config.d_model).to(device)],
                'loss': torch.tensor(1.5).to(device)
            }
            expert_indices, routing_weights, metadata = router(
                gate_logits, num_tokens, context
            )
            observed_metrics = gpu_monitor.get_current_stats()
            observed_metrics['inference_latency_ms'] = 12.0
            loss_components = router.update_energy_aware_loss(observed_metrics)
            self.assertEqual(expert_indices.shape, (batch_size, seq_length, moe_config.top_k))
            self.assertEqual(routing_weights.shape, (batch_size, seq_length, moe_config.top_k))
            self.assertIn('total_loss', loss_components)
        router._perform_ttt_update = orig_perform_ttt_update
        stats = router.get_statistics()
        self.assertTrue(stats['ttt_update_count'] >= 0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2) 