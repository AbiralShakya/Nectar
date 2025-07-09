#!/usr/bin/env python3
"""
Test script for network topology optimization with synthetic data.
Tests different expert placement strategies for data movement optimization.
"""

import torch
import torch.nn as nn
import argparse
import json
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.moe_models import NetworkTopologyOptimizer, DistributedMoELayer
from src.kernelcostmodel import KernelCostModel
from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class TopologyTestResult:
    """Results from network topology test."""
    placement_strategy: str
    avg_data_movement_gb: float
    avg_bandwidth_utilization: float
    avg_latency_ms: float
    avg_energy_joules: float
    load_balance_score: float
    thermal_balance_score: float

class NetworkTopologyTester:
    """
    Test network topology optimization with synthetic data.
    Tests different expert placement strategies.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        self.topology_optimizer = NetworkTopologyOptimizer(
            num_gpus=args.num_gpus,
            num_experts=args.num_experts
        )
        
        # TTT Router
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=2,
            lambda_energy=0.05
        ).to(self.device)
        
        # Load synthetic data
        self.synthetic_data = self._load_synthetic_data(args.synthetic_data_file)
        
        # Parse placement strategies
        self.placement_strategies = args.placement_strategies.split(',')
        
    def _load_synthetic_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load synthetic dataset."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data['data_points']
    
    def run_test(self) -> List[TopologyTestResult]:
        """Run comprehensive network topology test."""
        print("Running Network Topology Optimization Test")
        print(f"Placement Strategies: {self.placement_strategies}")
        
        results = []
        
        for strategy in self.placement_strategies:
            print(f"\nTesting placement strategy: {strategy}")
            
            result = self._test_placement_strategy(strategy)
            results.append(result)
            
            print(f"  Data Movement: {result.avg_data_movement_gb:.3f} GB")
            print(f"  Bandwidth Utilization: {result.avg_bandwidth_utilization:.3f}")
            print(f"  Latency: {result.avg_latency_ms:.2f} ms")
            print(f"  Energy: {result.avg_energy_joules:.6f} J")
            print(f"  Load Balance: {result.load_balance_score:.3f}")
            print(f"  Thermal Balance: {result.thermal_balance_score:.3f}")
        
        return results
    
    def _test_placement_strategy(self, strategy: str) -> TopologyTestResult:
        """Test a specific placement strategy."""
        # Configure topology optimizer
        self.topology_optimizer.set_placement_strategy(strategy)
        
        # Initialize metrics
        total_data_movement = 0.0
        total_bandwidth_util = 0.0
        total_latency = 0.0
        total_energy = 0.0
        load_balance_scores = []
        thermal_balance_scores = []
        
        num_batches = 0
        
        for batch_idx in range(self.args.num_batches):
            # Create synthetic input
            input_tensor = torch.randn(
                self.args.batch_size,
                self.args.seq_length,
                self.args.d_model
            ).to(self.device)
            
            # Get expert placement
            expert_placement = self.topology_optimizer.get_expert_placement()
            
            # Run routing
            start_time = time.time()
            routing_result = self._run_topology_aware_routing(input_tensor, expert_placement)
            end_time = time.time()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            data_movement_gb = self._estimate_data_movement(routing_result, expert_placement)
            bandwidth_util = self._estimate_bandwidth_utilization(routing_result, expert_placement)
            energy_joules = self._estimate_topology_energy(routing_result, expert_placement)
            
            # Calculate balance scores
            load_balance = self._calculate_load_balance(routing_result, expert_placement)
            thermal_balance = self._calculate_thermal_balance(expert_placement)
            
            # Accumulate metrics
            total_data_movement += data_movement_gb
            total_bandwidth_util += bandwidth_util
            total_latency += latency_ms
            total_energy += energy_joules
            load_balance_scores.append(load_balance)
            thermal_balance_scores.append(thermal_balance)
            num_batches += 1
            
            # Update topology optimizer
            self.topology_optimizer.update_placement_stats({
                'data_movement': data_movement_gb,
                'bandwidth_utilization': bandwidth_util,
                'load_balance': load_balance,
                'thermal_balance': thermal_balance
            })
        
        # Calculate averages
        avg_data_movement = total_data_movement / num_batches
        avg_bandwidth_util = total_bandwidth_util / num_batches
        avg_latency = total_latency / num_batches
        avg_energy = total_energy / num_batches
        avg_load_balance = np.mean(load_balance_scores)
        avg_thermal_balance = np.mean(thermal_balance_scores)
        
        return TopologyTestResult(
            placement_strategy=strategy,
            avg_data_movement_gb=avg_data_movement,
            avg_bandwidth_utilization=avg_bandwidth_util,
            avg_latency_ms=avg_latency,
            avg_energy_joules=avg_energy,
            load_balance_score=avg_load_balance,
            thermal_balance_score=avg_thermal_balance
        )
    
    def _run_topology_aware_routing(self, input_tensor: torch.Tensor,
                                  expert_placement: Dict[int, int]) -> Dict[str, Any]:
        """Run routing with topology awareness."""
        # Get routing decisions from TTT router
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={'expert_placement': expert_placement}
        )
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.args.num_experts, device=self.device)
        for i in range(self.args.num_experts):
            expert_usage[i] = (expert_indices == i).sum().item()
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'expert_usage': expert_usage,
            'expert_placement': expert_placement,
            'router_metadata': router_metadata
        }
    
    def _estimate_data_movement(self, routing_result: Dict[str, Any],
                              expert_placement: Dict[int, int]) -> float:
        """Estimate data movement in GB."""
        expert_usage = routing_result['expert_usage']
        batch_size = routing_result['expert_indices'].size(0)
        seq_length = routing_result['expert_indices'].size(1)
        
        # Calculate data movement based on expert placement
        total_movement = 0.0
        
        for expert_id, gpu_id in expert_placement.items():
            if expert_usage[expert_id] > 0:
                # Estimate data size per expert (rough approximation)
                data_size_gb = (expert_usage[expert_id] * seq_length * self.args.d_model * 4) / (1024**3)  # 4 bytes per float
                total_movement += data_size_gb
        
        return total_movement
    
    def _estimate_bandwidth_utilization(self, routing_result: Dict[str, Any],
                                      expert_placement: Dict[int, int]) -> float:
        """Estimate bandwidth utilization."""
        # Simple bandwidth estimation based on data movement
        data_movement_gb = self._estimate_data_movement(routing_result, expert_placement)
        
        # Assume 100 GB/s bandwidth for A100
        max_bandwidth_gb_s = 100.0
        
        # Estimate time for data movement
        movement_time_s = data_movement_gb / max_bandwidth_gb_s
        
        # Estimate total computation time (rough approximation)
        total_tokens = routing_result['expert_indices'].numel()
        computation_time_s = total_tokens * 0.000001  # Rough estimate
        
        # Bandwidth utilization
        if movement_time_s + computation_time_s > 0:
            utilization = movement_time_s / (movement_time_s + computation_time_s)
        else:
            utilization = 0.0
        
        return min(1.0, utilization)
    
    def _estimate_topology_energy(self, routing_result: Dict[str, Any],
                                expert_placement: Dict[int, int]) -> float:
        """Estimate energy consumption considering topology."""
        batch_size = routing_result['expert_indices'].size(0)
        seq_length = routing_result['expert_indices'].size(1)
        
        # Base routing energy
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
        
        # Data movement energy penalty
        data_movement_gb = self._estimate_data_movement(routing_result, expert_placement)
        movement_energy = data_movement_gb * 0.1  # Rough estimate: 0.1 J per GB
        
        total_energy = routing_energy + expert_energy + movement_energy
        
        return total_energy
    
    def _calculate_load_balance(self, routing_result: Dict[str, Any],
                              expert_placement: Dict[int, int]) -> float:
        """Calculate load balance score."""
        expert_usage = routing_result['expert_usage']
        total_usage = expert_usage.sum().item()
        
        if total_usage == 0:
            return 0.0
        
        # Calculate per-GPU load
        gpu_loads = {}
        for expert_id, gpu_id in expert_placement.items():
            if gpu_id not in gpu_loads:
                gpu_loads[gpu_id] = 0
            gpu_loads[gpu_id] += expert_usage[expert_id].item()
        
        # Calculate load balance
        loads = list(gpu_loads.values())
        if len(loads) > 1:
            load_variance = np.var(loads)
            expected_load = np.mean(loads)
            if expected_load > 0:
                balance_score = 1.0 / (1.0 + load_variance / (expected_load ** 2))
            else:
                balance_score = 0.0
        else:
            balance_score = 1.0
        
        return min(1.0, max(0.0, balance_score))
    
    def _calculate_thermal_balance(self, expert_placement: Dict[int, int]) -> float:
        """Calculate thermal balance score."""
        # Simple thermal balance based on expert distribution
        gpu_expert_counts = {}
        for expert_id, gpu_id in expert_placement.items():
            if gpu_id not in gpu_expert_counts:
                gpu_expert_counts[gpu_id] = 0
            gpu_expert_counts[gpu_id] += 1
        
        if len(gpu_expert_counts) > 1:
            expert_counts = list(gpu_expert_counts.values())
            count_variance = np.var(expert_counts)
            expected_count = np.mean(expert_counts)
            if expected_count > 0:
                thermal_balance = 1.0 / (1.0 + count_variance / (expected_count ** 2))
            else:
                thermal_balance = 0.0
        else:
            thermal_balance = 1.0
        
        return min(1.0, max(0.0, thermal_balance))
    
    def save_results(self, results: List[TopologyTestResult], output_file: str):
        """Save test results to file."""
        output_data = {
            'test_config': {
                'num_gpus': self.args.num_gpus,
                'num_experts': self.args.num_experts,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'num_batches': self.args.num_batches,
                'placement_strategies': self.placement_strategies
            },
            'results': [
                {
                    'placement_strategy': r.placement_strategy,
                    'avg_data_movement_gb': r.avg_data_movement_gb,
                    'avg_bandwidth_utilization': r.avg_bandwidth_utilization,
                    'avg_latency_ms': r.avg_latency_ms,
                    'avg_energy_joules': r.avg_energy_joules,
                    'load_balance_score': r.load_balance_score,
                    'thermal_balance_score': r.thermal_balance_score
                }
                for r in results
            ],
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[TopologyTestResult]):
        """Print test summary."""
        print("\n=== Network Topology Test Summary ===")
        
        for result in results:
            print(f"\nStrategy: {result.placement_strategy}")
            print(f"  Data Movement: {result.avg_data_movement_gb:.3f} GB")
            print(f"  Bandwidth Utilization: {result.avg_bandwidth_utilization:.3f}")
            print(f"  Latency: {result.avg_latency_ms:.2f} ms")
            print(f"  Energy: {result.avg_energy_joules:.6f} J")
            print(f"  Load Balance: {result.load_balance_score:.3f}")
            print(f"  Thermal Balance: {result.thermal_balance_score:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Test Network Topology Optimization")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=75, help="Number of batches")
    parser.add_argument("--placement_strategies", type=str, default="load_balanced,bandwidth_optimized,thermal_aware",
                       help="Comma-separated placement strategies")
    parser.add_argument("--synthetic_data_file", type=str, required=True, help="Synthetic data file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    
    args = parser.parse_args()
    
    print("=== Network Topology Optimization Test ===")
    print(f"Placement Strategies: {args.placement_strategies}")
    
    # Run test
    tester = NetworkTopologyTester(args)
    results = tester.run_test()
    
    # Save results
    tester.save_results(results, args.output_file)
    
    # Print summary
    tester.print_summary(results)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 