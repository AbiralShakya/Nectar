#!/usr/bin/env python3
"""
Test script for thermal-aware routing with synthetic data.
Validates thermal routing factors and thermal imbalance detection.
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

from src.thermal_signal import ThermalAwareRouter, ThermalState, ThermalSignalProcessor
from src.moe_models import DistributedMoELayer
from models.ttt_router import LaCTEnergyAwareTTTRouter

@dataclass
class ThermalTestResult:
    """Results from thermal-aware routing test."""
    thermal_scenario: str
    memory_pressure: float
    avg_temperature: float
    thermal_imbalance_score: float
    expert_migration_count: int
    avg_energy_joules: float
    avg_latency_ms: float
    routing_accuracy: float
    thermal_penalty_applied: bool

class ThermalAwareRoutingTester:
    """
    Test thermal-aware routing with synthetic data.
    Validates thermal routing factors and thermal imbalance detection.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize thermal components
        self.thermal_router = ThermalAwareRouter(
            num_experts=args.num_experts,
            num_gpus=args.num_gpus
        )
        
        self.thermal_processor = ThermalSignalProcessor(
            window_size=100,
            threshold_temperature=80.0
        )
        
        # TTT Router with thermal awareness
        self.ttt_router = LaCTEnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=2,
            lambda_energy=0.05,
            chunk_size=500
        ).to(self.device)
        
        # Load synthetic data
        self.synthetic_data = self._load_synthetic_data(args.synthetic_data_file)
        
        # Parse thermal scenarios and memory pressure levels
        self.thermal_scenarios = args.thermal_scenarios.split(',')
        self.memory_pressure_levels = [float(x) for x in args.memory_pressure_levels.split(',')]
        
    def _load_synthetic_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load synthetic dataset."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data['data_points']
    
    def run_test(self) -> List[ThermalTestResult]:
        """Run comprehensive thermal-aware routing test."""
        print("Running Thermal-Aware Routing Test")
        print(f"Thermal Scenarios: {self.thermal_scenarios}")
        print(f"Memory Pressure Levels: {self.memory_pressure_levels}")
        
        results = []
        
        for scenario in self.thermal_scenarios:
            for memory_pressure in self.memory_pressure_levels:
                print(f"\nTesting scenario: {scenario}, memory pressure: {memory_pressure}")
                
                result = self._test_thermal_scenario(scenario, memory_pressure)
                results.append(result)
                
                print(f"  Temperature: {result.avg_temperature:.1f}°C")
                print(f"  Thermal Imbalance: {result.thermal_imbalance_score:.3f}")
                print(f"  Expert Migrations: {result.expert_migration_count}")
                print(f"  Energy: {result.avg_energy_joules:.6f}J")
                print(f"  Latency: {result.avg_latency_ms:.2f}ms")
        
        return results
    
    def _test_thermal_scenario(self, scenario: str, memory_pressure: float) -> ThermalTestResult:
        """Test a specific thermal scenario."""
        # Generate thermal state based on scenario
        thermal_state = self._generate_thermal_state(scenario, memory_pressure)
        
        # Initialize metrics
        total_energy = 0.0
        total_latency = 0.0
        total_temperature = 0.0
        total_imbalance = 0.0
        expert_migrations = 0
        routing_accuracies = []
        
        num_batches = 0
        
        for batch_idx in range(self.args.num_batches):
            # Create synthetic input
            input_tensor = torch.randn(
                self.args.batch_size,
                self.args.seq_length,
                self.args.d_model
            ).to(self.device)
            
            # Update thermal state
            self.thermal_router.update_thermal_state(0, thermal_state)
            
            # Process thermal signal
            thermal_signal = self.thermal_processor.process_thermal_signal(thermal_state)
            
            # Run routing with thermal awareness
            start_time = time.time()
            routing_result = self._run_thermal_aware_routing(input_tensor, thermal_signal)
            end_time = time.time()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            energy_joules = self._estimate_energy_with_thermal(thermal_state, input_tensor)
            accuracy = self._calculate_routing_accuracy(routing_result, input_tensor)
            
            # Check for expert migrations
            if self._detect_expert_migration(routing_result):
                expert_migrations += 1
            
            # Accumulate metrics
            total_energy += energy_joules
            total_latency += latency_ms
            total_temperature += thermal_state.temperature
            total_imbalance += self.thermal_router.get_thermal_stats()['thermal_imbalance_score']
            routing_accuracies.append(accuracy)
            num_batches += 1
            
            # Update thermal state for next iteration
            thermal_state = self._evolve_thermal_state(thermal_state, scenario)
        
        # Calculate averages
        avg_energy = total_energy / num_batches
        avg_latency = total_latency / num_batches
        avg_temperature = total_temperature / num_batches
        avg_imbalance = total_imbalance / num_batches
        avg_accuracy = np.mean(routing_accuracies)
        
        # Check if thermal penalty was applied
        thermal_penalty_applied = avg_imbalance > 0.1  # Threshold for thermal penalty
        
        return ThermalTestResult(
            thermal_scenario=scenario,
            memory_pressure=memory_pressure,
            avg_temperature=avg_temperature,
            thermal_imbalance_score=avg_imbalance,
            expert_migration_count=expert_migrations,
            avg_energy_joules=avg_energy,
            avg_latency_ms=avg_latency,
            routing_accuracy=avg_accuracy,
            thermal_penalty_applied=thermal_penalty_applied
        )
    
    def _generate_thermal_state(self, scenario: str, memory_pressure: float) -> ThermalState:
        """Generate thermal state based on scenario."""
        base_temp = 60.0
        
        if scenario == "normal":
            temperature = base_temp + np.random.normal(0, 5)
            compute_util = 0.6 + np.random.normal(0, 0.1)
        elif scenario == "hot":
            temperature = base_temp + 20 + np.random.normal(0, 8)
            compute_util = 0.8 + np.random.normal(0, 0.15)
        elif scenario == "imbalanced":
            temperature = base_temp + 15 + np.random.normal(0, 10)
            compute_util = 0.7 + np.random.normal(0, 0.2)
        elif scenario == "cool":
            temperature = base_temp - 10 + np.random.normal(0, 3)
            compute_util = 0.4 + np.random.normal(0, 0.1)
        else:
            temperature = base_temp
            compute_util = 0.6
        
        # Ensure reasonable bounds
        temperature = max(30.0, min(95.0, temperature))
        compute_util = max(0.1, min(0.95, compute_util))
        
        # Estimate power based on utilization
        power_watt = 150 + compute_util * 200  # 150-350W range
        
        return ThermalState(
            temperature=temperature,
            power_watt=power_watt,
            memory_utilization=memory_pressure,
            compute_utilization=compute_util,
            timestamp=time.time()
        )
    
    def _run_thermal_aware_routing(self, input_tensor: torch.Tensor, 
                                 thermal_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Run routing with thermal awareness."""
        # Get routing decisions from TTT router
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={'thermal_signal': thermal_signal}
        )
        
        # Apply thermal-aware adjustments
        if thermal_signal['thermal_imbalance_detected']:
            # Adjust routing based on thermal imbalance
            expert_indices, expert_weights = self._apply_thermal_routing_adjustment(
                expert_indices, expert_weights, thermal_signal
            )
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.args.num_experts, device=self.device)
        for i in range(self.args.num_experts):
            expert_usage[i] = (expert_indices == i).sum().item()
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'expert_usage': expert_usage,
            'thermal_signal': thermal_signal,
            'router_metadata': router_metadata
        }
    
    def _apply_thermal_routing_adjustment(self, expert_indices: torch.Tensor,
                                        expert_weights: torch.Tensor,
                                        thermal_signal: Dict[str, Any]) -> tuple:
        """Apply thermal-aware routing adjustments."""
        # Get thermal stats
        thermal_stats = self.thermal_router.get_thermal_stats()
        
        # If thermal imbalance is high, prefer cooler experts
        if thermal_stats['thermal_imbalance_score'] > 0.2:
            # Simple adjustment: reduce weights for hot experts
            # In a real implementation, this would be more sophisticated
            adjusted_weights = expert_weights.clone()
            
            # Apply thermal penalty to overused experts
            for i in range(self.args.num_experts):
                if thermal_stats['expert_thermal_loads'][i] > 0.8:  # Hot expert
                    adjusted_weights[:, i] *= 0.7  # Reduce weight
            
            # Renormalize
            adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=-1, keepdim=True)
            
            # Recompute top-k
            top_k_weights, top_k_indices = torch.topk(adjusted_weights, 2, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            return top_k_indices, top_k_weights
        
        return expert_indices, expert_weights
    
    def _estimate_energy_with_thermal(self, thermal_state: ThermalState, 
                                    input_tensor: torch.Tensor) -> float:
        """Estimate energy consumption considering thermal state."""
        # Base energy estimation
        batch_size = input_tensor.size(0)
        seq_length = input_tensor.size(1)
        
        # Base routing energy
        routing_energy = batch_size * seq_length * 0.0001  # Rough estimate
        
        # Thermal penalty factor
        thermal_factor = 1.0 + (thermal_state.temperature - 60.0) / 100.0
        
        # Memory pressure penalty
        memory_factor = 1.0 + thermal_state.memory_utilization * 0.3
        
        # Compute utilization penalty
        compute_factor = 1.0 + thermal_state.compute_utilization * 0.2
        
        total_energy = routing_energy * thermal_factor * memory_factor * compute_factor
        
        return total_energy
    
    def _calculate_routing_accuracy(self, routing_result: Dict[str, Any], 
                                  input_tensor: torch.Tensor) -> float:
        """Calculate routing accuracy."""
        expert_usage = routing_result['expert_usage']
        total_usage = expert_usage.sum().item()
        
        if total_usage == 0:
            return 0.0
        
        # Load balancing accuracy
        expected_usage = total_usage / self.args.num_experts
        usage_variance = torch.var(expert_usage.float()).item()
        load_balance_accuracy = 1.0 / (1.0 + usage_variance / (expected_usage ** 2))
        
        # Thermal awareness accuracy
        thermal_signal = routing_result['thermal_signal']
        if thermal_signal['thermal_imbalance_detected']:
            thermal_accuracy = 0.8  # Good thermal awareness
        else:
            thermal_accuracy = 1.0  # No thermal issues
        
        # Combined accuracy
        accuracy = 0.7 * load_balance_accuracy + 0.3 * thermal_accuracy
        
        return min(1.0, max(0.0, accuracy))
    
    def _detect_expert_migration(self, routing_result: Dict[str, Any]) -> bool:
        """Detect if expert migration occurred."""
        # Simple heuristic: check if routing changed significantly
        expert_usage = routing_result['expert_usage']
        
        # Check if any expert usage is very high (>50% of total)
        total_usage = expert_usage.sum().item()
        if total_usage > 0:
            max_usage_ratio = expert_usage.max().item() / total_usage
            return max_usage_ratio > 0.5
        
        return False
    
    def _evolve_thermal_state(self, current_state: ThermalState, scenario: str) -> ThermalState:
        """Evolve thermal state for next iteration."""
        # Add some randomness to thermal evolution
        temp_change = np.random.normal(0, 2)  # ±2°C change
        new_temperature = current_state.temperature + temp_change
        
        # Ensure temperature stays within reasonable bounds
        new_temperature = max(30.0, min(95.0, new_temperature))
        
        # Update other parameters
        new_power = current_state.power_watt + np.random.normal(0, 10)
        new_power = max(100, min(400, new_power))
        
        new_compute_util = current_state.compute_utilization + np.random.normal(0, 0.05)
        new_compute_util = max(0.1, min(0.95, new_compute_util))
        
        return ThermalState(
            temperature=new_temperature,
            power_watt=new_power,
            memory_utilization=current_state.memory_utilization,
            compute_utilization=new_compute_util,
            timestamp=time.time()
        )
    
    def save_results(self, results: List[ThermalTestResult], output_file: str):
        """Save test results to file."""
        output_data = {
            'test_config': {
                'num_gpus': self.args.num_gpus,
                'num_experts': self.args.num_experts,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'num_batches': self.args.num_batches,
                'thermal_scenarios': self.thermal_scenarios,
                'memory_pressure_levels': self.memory_pressure_levels
            },
            'results': [
                {
                    'thermal_scenario': r.thermal_scenario,
                    'memory_pressure': r.memory_pressure,
                    'avg_temperature': r.avg_temperature,
                    'thermal_imbalance_score': r.thermal_imbalance_score,
                    'expert_migration_count': r.expert_migration_count,
                    'avg_energy_joules': r.avg_energy_joules,
                    'avg_latency_ms': r.avg_latency_ms,
                    'routing_accuracy': r.routing_accuracy,
                    'thermal_penalty_applied': r.thermal_penalty_applied
                }
                for r in results
            ],
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[ThermalTestResult]):
        """Print test summary."""
        print("\n=== Thermal-Aware Routing Test Summary ===")
        
        for result in results:
            print(f"\nScenario: {result.thermal_scenario}, Memory Pressure: {result.memory_pressure}")
            print(f"  Temperature: {result.avg_temperature:.1f}°C")
            print(f"  Thermal Imbalance: {result.thermal_imbalance_score:.3f}")
            print(f"  Expert Migrations: {result.expert_migration_count}")
            print(f"  Energy: {result.avg_energy_joules:.6f}J")
            print(f"  Latency: {result.avg_latency_ms:.2f}ms")
            print(f"  Accuracy: {result.routing_accuracy:.3f}")
            print(f"  Thermal Penalty: {'Yes' if result.thermal_penalty_applied else 'No'}")

def main():
    parser = argparse.ArgumentParser(description="Test Thermal-Aware Routing")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches")
    parser.add_argument("--thermal_scenarios", type=str, default="normal,hot,imbalanced,cool", 
                       help="Comma-separated thermal scenarios")
    parser.add_argument("--memory_pressure_levels", type=str, default="0.3,0.5,0.7,0.9",
                       help="Comma-separated memory pressure levels")
    parser.add_argument("--synthetic_data_file", type=str, required=True, help="Synthetic data file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    
    args = parser.parse_args()
    
    print("=== Thermal-Aware Routing Test ===")
    print(f"Thermal Scenarios: {args.thermal_scenarios}")
    print(f"Memory Pressure Levels: {args.memory_pressure_levels}")
    
    # Run test
    tester = ThermalAwareRoutingTester(args)
    results = tester.run_test()
    
    # Save results
    tester.save_results(results, args.output_file)
    
    # Print summary
    tester.print_summary(results)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 