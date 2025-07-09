#!/usr/bin/env python3
"""
Inline test script for energy-aware TTT routing with synthetic data.
Generates synthetic data inline to avoid file dependency issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not available - plots will be skipped")

from src.moe_models import DistributedMoELayer, NetworkTopologyOptimizer
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState
from models.ttt_router import EnergyAwareTTTRouter  # Use existing class

@dataclass
class TTTTestResult:
    """Results from TTT routing test."""
    lambda_energy: float
    batch_size: int
    seq_length: int
    num_experts: int
    moe_top_k: int
    avg_energy_joules: float
    avg_latency_ms: float
    avg_power_watt: float
    avg_accuracy: float
    thermal_imbalance_score: float
    routing_entropy: float
    expert_usage_distribution: List[float]
    ttt_update_count: int
    energy_savings_percent: float
    accuracy_loss_percent: float

class EnergyAwareTTTSyntheticTester:
    """
    Test energy-aware TTT routing with inline synthetic data.
    Validates routing factors for energy and thermal optimization.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        self.gpu_monitor = GpuSystemMonitor()
        
        # TTT Router - Use EnergyAwareTTTRouter
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=args.moe_top_k,
            lambda_energy=args.lambda_energy
        ).to(self.device)
        
        # Thermal router
        if args.enable_thermal_awareness:
            self.thermal_router = ThermalAwareRouter(
                num_experts=args.num_experts,
                num_gpus=1
            )
        
        # Generate synthetic data inline
        self.synthetic_data = self._generate_synthetic_data()
        
        # Results storage
        self.results = []
        
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic data inline."""
        print("Generating synthetic data inline...")
        
        data_points = []
        for i in range(1000):  # Generate 1000 synthetic data points
            # Create synthetic patterns
            patterns = []
            for j in range(self.args.batch_size):
                # Create realistic patterns for MoE routing
                pattern = np.random.normal(0, 1, self.args.d_model)
                # Add some structure to make it more realistic
                pattern[:self.args.d_model//4] += np.random.normal(0, 0.5)  # Add bias to first quarter
                patterns.append(pattern.tolist())
            
            data_point = {
                'id': i,
                'patterns': patterns,
                'metadata': {
                    'batch_size': self.args.batch_size,
                    'seq_length': self.args.seq_length,
                    'd_model': self.args.d_model
                }
            }
            data_points.append(data_point)
        
        print(f"Generated {len(data_points)} synthetic data points")
        return data_points
    
    def run_test(self) -> TTTTestResult:
        """Run comprehensive TTT routing test."""
        print(f"Running Energy-Aware TTT Test with lambda_energy={self.args.lambda_energy}")
        
        # Test metrics
        total_energy = 0.0
        total_latency = 0.0
        total_power = 0.0
        total_accuracy = 0.0
        routing_entropies = []
        expert_usage_counts = torch.zeros(self.args.num_experts, device=self.device)
        thermal_imbalance_scores = []
        
        num_batches = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            for batch_idx in range(self.args.num_batches):
                # Get synthetic batch
                batch_data = self.synthetic_data[batch_idx % len(self.synthetic_data)]
                
                # Create input tensor from synthetic data
                input_tensor = torch.randn(
                    self.args.batch_size, 
                    self.args.seq_length, 
                    self.args.d_model
                ).to(self.device)
                
                # Apply patterns from synthetic data
                if 'patterns' in batch_data:
                    patterns = batch_data['patterns']
                    for i, pattern in enumerate(patterns[:self.args.batch_size]):
                        if i < input_tensor.size(0):
                            input_tensor[i] += torch.tensor(pattern, device=self.device) * 0.1
                
                # Apply noise if enabled
                if self.args.enable_noise_injection:
                    input_tensor = self._apply_noise(input_tensor, self.args.noise_level)
                
                # Measure baseline performance
                baseline_metrics = self._measure_baseline_performance(input_tensor)
                
                # Run TTT routing
                start_time = time.time()
                routing_result = self._run_ttt_routing(input_tensor)
                end_time = time.time()
                
                # Calculate metrics
                latency_ms = (end_time - start_time) * 1000
                energy_joules = self._estimate_energy_consumption(routing_result, input_tensor)
                power_watt = energy_joules / (latency_ms / 1000)
                accuracy = self._calculate_accuracy(routing_result, input_tensor)
                
                # Update thermal state if enabled
                if self.args.enable_thermal_awareness:
                    thermal_state = self._get_thermal_state()
                    self.thermal_router.update_thermal_state(0, thermal_state)
                    thermal_imbalance = self.thermal_router.get_thermal_stats()['thermal_imbalance_score']
                    thermal_imbalance_scores.append(thermal_imbalance)
                
                # Update TTT router
                feedback = {
                    'estimated_energy': energy_joules,
                    'expert_usage': routing_result['expert_usage'],
                    'token_count': input_tensor.numel(),
                    'batch_size': input_tensor.size(0),
                    'seq_length': input_tensor.size(1)
                }
                self.ttt_router.ttt_update(feedback)
                
                # Accumulate metrics
                total_energy += energy_joules
                total_latency += latency_ms
                total_power += power_watt
                total_accuracy += accuracy
                routing_entropies.append(routing_result['routing_entropy'])
                expert_usage_counts += routing_result['expert_usage']
                num_batches += 1
                
                # Print progress
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}: Energy={energy_joules:.6f}J, "
                          f"Latency={latency_ms:.2f}ms, Power={power_watt:.1f}W, "
                          f"Accuracy={accuracy:.3f}")
        
        # Calculate averages
        avg_energy = total_energy / num_batches
        avg_latency = total_latency / num_batches
        avg_power = total_power / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_routing_entropy = float(np.mean(routing_entropies))
        expert_usage_distribution = (expert_usage_counts / expert_usage_counts.sum()).tolist()
        avg_thermal_imbalance = float(np.mean(thermal_imbalance_scores)) if thermal_imbalance_scores else 0.0
        
        # Calculate improvements
        baseline_energy = self._get_baseline_energy()
        baseline_accuracy = 0.95  # Assume baseline accuracy
        energy_savings = ((baseline_energy - avg_energy) / baseline_energy) * 100
        accuracy_loss = ((baseline_accuracy - avg_accuracy) / baseline_accuracy) * 100
        
        result = TTTTestResult(
            lambda_energy=self.args.lambda_energy,
            batch_size=self.args.batch_size,
            seq_length=self.args.seq_length,
            num_experts=self.args.num_experts,
            moe_top_k=self.args.moe_top_k,
            avg_energy_joules=avg_energy,
            avg_latency_ms=avg_latency,
            avg_power_watt=avg_power,
            avg_accuracy=avg_accuracy,
            thermal_imbalance_score=avg_thermal_imbalance,
            routing_entropy=avg_routing_entropy,
            expert_usage_distribution=expert_usage_distribution,
            ttt_update_count=self.ttt_router.ttt_update_count,
            energy_savings_percent=energy_savings,
            accuracy_loss_percent=accuracy_loss
        )
        
        return result
    
    def _run_ttt_routing(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run TTT routing on input tensor."""
        # Get routing decisions
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={'batch_size': input_tensor.size(0)}
        )
        
        # Calculate routing entropy
        routing_entropy = self._calculate_routing_entropy(expert_weights)
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.args.num_experts, device=self.device)
        for i in range(self.args.num_experts):
            expert_usage[i] = (expert_indices == i).sum().item()
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'router_metadata': router_metadata,
            'routing_entropy': routing_entropy,
            'expert_usage': expert_usage
        }
    
    def _measure_baseline_performance(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """Measure baseline performance without TTT."""
        # Simple baseline measurement
        start_time = time.time()
        with torch.no_grad():
            _ = self.ttt_router.gate(input_tensor)
        end_time = time.time()
        
        return {
            'latency_ms': (end_time - start_time) * 1000,
            'energy_joules': input_tensor.numel() * 0.0001  # Rough estimate
        }
    
    def _estimate_energy_consumption(self, routing_result: Dict[str, Any], 
                                   input_tensor: torch.Tensor) -> float:
        """Estimate energy consumption for routing result."""
        # Use kernel cost model to estimate energy
        batch_size = input_tensor.size(0)
        
        # Routing energy
        routing_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        routing_energy = routing_cost["energy_joules"]
        
        # Expert computation energy
        expert_usage = routing_result['expert_usage']
        total_expert_tokens = expert_usage.sum().item()
        
        if total_expert_tokens > 0:
            expert_cost = self.kernel_cost_model.get_cost("ffn_gate", int(total_expert_tokens))
            expert_energy = expert_cost["energy_joules"]
        else:
            expert_energy = 0.0
        
        # Add thermal penalty if enabled
        if self.args.enable_thermal_awareness:
            thermal_stats = self.thermal_router.get_thermal_stats()
            thermal_factor = 1.0 + thermal_stats['thermal_imbalance_score'] * 0.2
            routing_energy *= thermal_factor
            expert_energy *= thermal_factor
        
        return routing_energy + expert_energy
    
    def _calculate_accuracy(self, routing_result: Dict[str, Any], 
                          input_tensor: torch.Tensor) -> float:
        """Calculate routing accuracy."""
        # Simple accuracy metric based on expert usage distribution
        expert_usage = routing_result['expert_usage']
        total_usage = expert_usage.sum().item()
        
        if total_usage == 0:
            return 0.0
        
        # Calculate load balancing accuracy
        expected_usage = total_usage / self.args.num_experts
        usage_variance = torch.var(expert_usage.float()).item()
        load_balance_accuracy = 1.0 / (1.0 + usage_variance / (expected_usage ** 2))
        
        # Calculate routing confidence
        expert_weights = routing_result['expert_weights']
        routing_confidence = torch.mean(expert_weights).item()
        
        # Combined accuracy
        accuracy = 0.7 * load_balance_accuracy + 0.3 * routing_confidence
        
        return min(1.0, max(0.0, accuracy))
    
    def _calculate_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Calculate routing entropy."""
        # Add small epsilon to avoid log(0)
        weights = expert_weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return torch.mean(entropy).item()
    
    def _apply_noise(self, tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Apply synthetic noise to tensor."""
        if noise_level == 0:
            return tensor
        
        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise
    
    def _get_thermal_state(self) -> ThermalState:
        """Get current thermal state."""
        gpu_stats = self.gpu_monitor.get_current_stats()
        
        return ThermalState(
            temperature=gpu_stats.get('temperature', 60.0),
            power_watt=gpu_stats.get('power_watt', 200.0),
            memory_utilization=gpu_stats.get('memory_utilization_percent', 0.6) / 100.0,
            compute_utilization=gpu_stats.get('gpu_utilization_percent', 0.7) / 100.0,
            timestamp=time.time()
        )
    
    def _get_baseline_energy(self) -> float:
        """Get baseline energy consumption."""
        # Estimate baseline energy using the same routing logic as actual energy
        batch_size = self.args.batch_size
        seq_length = self.args.seq_length
        total_tokens = batch_size * seq_length
        
        # Routing energy (same for baseline and actual)
        routing_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        routing_energy = routing_cost["energy_joules"]
        
        # Expert energy: only count tokens that are actually routed (top-k selection)
        # With top-k=2 and num_experts=16, each token goes to 2/16 = 1/8 of experts
        # So total expert tokens = total_tokens * top_k / num_experts
        expert_tokens = total_tokens * self.args.moe_top_k / self.args.num_experts
        expert_cost = self.kernel_cost_model.get_cost("ffn_gate", int(expert_tokens))
        expert_energy = expert_cost["energy_joules"]
        
        return routing_energy + expert_energy
    
    def save_results(self, result: TTTTestResult, output_file: str):
        """Save test results to file."""
        output_data = {
            'test_config': {
                'lambda_energy': self.args.lambda_energy,
                'num_experts': self.args.num_experts,
                'moe_top_k': self.args.moe_top_k,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'enable_thermal_awareness': self.args.enable_thermal_awareness,
                'enable_noise_injection': self.args.enable_noise_injection,
                'noise_level': self.args.noise_level,
                'error_margin': self.args.error_margin
            },
            'results': {
                'avg_energy_joules': float(result.avg_energy_joules),
                'avg_latency_ms': float(result.avg_latency_ms),
                'avg_power_watt': float(result.avg_power_watt),
                'avg_accuracy': float(result.avg_accuracy),
                'thermal_imbalance_score': float(result.thermal_imbalance_score),
                'routing_entropy': float(result.routing_entropy),
                'expert_usage_distribution': [float(x) for x in result.expert_usage_distribution],
                'ttt_update_count': int(result.ttt_update_count),
                'energy_savings_percent': float(result.energy_savings_percent),
                'accuracy_loss_percent': float(result.accuracy_loss_percent)
            },
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, result: TTTTestResult):
        """Print test summary."""
        print("\n=== Energy-Aware TTT Test Summary ===")
        print(f"Lambda Energy: {result.lambda_energy}")
        print(f"Batch Size: {result.batch_size}")
        print(f"Sequence Length: {result.seq_length}")
        print(f"Number of Experts: {result.num_experts}")
        print(f"MoE Top-K: {result.moe_top_k}")
        print()
        print("Performance Metrics:")
        print(f"  Average Energy: {result.avg_energy_joules:.6f} J")
        print(f"  Average Latency: {result.avg_latency_ms:.2f} ms")
        print(f"  Average Power: {result.avg_power_watt:.1f} W")
        print(f"  Average Accuracy: {result.avg_accuracy:.3f}")
        print(f"  Thermal Imbalance Score: {result.thermal_imbalance_score:.3f}")
        print(f"  Routing Entropy: {result.routing_entropy:.3f}")
        print()
        print("Optimization Results:")
        print(f"  Energy Savings: {result.energy_savings_percent:.2f}%")
        print(f"  Accuracy Loss: {result.accuracy_loss_percent:.2f}%")
        print(f"  TTT Updates: {result.ttt_update_count}")
        print()
        print("Expert Usage Distribution:")
        for i, usage in enumerate(result.expert_usage_distribution):
            print(f"  Expert {i}: {usage:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Test Energy-Aware TTT with Inline Synthetic Data")
    parser.add_argument("--lambda_energy", type=float, default=0.05, help="Energy penalty weight")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=2, help="Top-k experts to route to")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--synthetic_data_file", type=str, default="", help="Synthetic data file (not used in inline version)")
    parser.add_argument("--enable_thermal_awareness", action="store_true", help="Enable thermal awareness")
    parser.add_argument("--enable_noise_injection", action="store_true", help="Enable noise injection")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level")
    parser.add_argument("--error_margin", type=float, default=0.1, help="Error margin")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    
    args = parser.parse_args()
    
    print("=== Energy-Aware TTT Synthetic Test (Inline) ===")
    print(f"Lambda Energy: {args.lambda_energy}")
    print(f"Thermal Awareness: {args.enable_thermal_awareness}")
    print(f"Noise Injection: {args.enable_noise_injection}")
    print(f"Noise Level: {args.noise_level}")
    print(f"Error Margin: {args.error_margin}")
    
    # Run test
    tester = EnergyAwareTTTSyntheticTester(args)
    result = tester.run_test()
    
    # Save results
    tester.save_results(result, args.output_file)
    
    # Print summary
    tester.print_summary(result)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 