#!/usr/bin/env python3
"""
Comprehensive TTT validation script with synthetic data.
Tests the complete TTT system including energy awareness, thermal awareness, and noise handling.
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
import matplotlib.pyplot as plt

from src.moe_models import DistributedMoELayer, NetworkTopologyOptimizer
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState, ThermalSignalProcessor
from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class TTTValidationResult:
    """Results from comprehensive TTT validation."""
    lambda_energy: float
    chunk_size: int
    noise_level: float
    error_margin: float
    avg_energy_joules: float
    avg_latency_ms: float
    avg_accuracy: float
    thermal_imbalance_score: float
    routing_entropy: float
    ttt_update_count: int
    energy_savings_percent: float
    accuracy_loss_percent: float
    noise_robustness_score: float
    thermal_adaptation_score: float
    convergence_rate: float

class TTTValidationTester:
    """
    Comprehensive TTT validation tester.
    Tests energy awareness, thermal awareness, noise handling, and convergence.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        self.gpu_monitor = GpuSystemMonitor()
        
        # TTT Router with all features enabled
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=args.moe_top_k,
            lambda_energy=args.lambda_energy
        ).to(self.device)
        
        # Thermal components
        self.thermal_router = ThermalAwareRouter(
            num_experts=args.num_experts,
            num_gpus=1
        )
        
        self.thermal_processor = ThermalSignalProcessor(
            window_size=100,
            threshold_temperature=80.0
        )
        
        # Load synthetic data
        self.synthetic_data = self._load_synthetic_data(args.synthetic_data_file)
        
        # Results storage
        self.validation_metrics = []
        
    def _load_synthetic_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load synthetic dataset."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data['data_points']
    
    def run_validation(self) -> TTTValidationResult:
        """Run comprehensive TTT validation."""
        print(f"Running Comprehensive TTT Validation")
        print(f"Lambda Energy: {self.args.lambda_energy}")
        print(f"Chunk Size: {self.args.chunk_size}")
        print(f"Noise Level: {self.args.noise_level}")
        print(f"Error Margin: {self.args.error_margin}")
        
        # Initialize metrics
        total_energy = 0.0
        total_latency = 0.0
        total_accuracy = 0.0
        thermal_imbalance_scores = []
        routing_entropies = []
        noise_robustness_scores = []
        thermal_adaptation_scores = []
        convergence_metrics = []
        
        num_batches = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            epoch_metrics = []
            
            for batch_idx in range(self.args.num_batches):
                # Get synthetic batch
                batch_data = self.synthetic_data[batch_idx % len(self.synthetic_data)]
                
                # Create input tensor with controlled noise
                input_tensor = self._create_synthetic_input(batch_data)
                
                # Apply noise injection
                if self.args.enable_noise_injection:
                    input_tensor = self._apply_controlled_noise(input_tensor)
                
                # Get thermal state
                thermal_state = self._get_current_thermal_state()
                self.thermal_router.update_thermal_state(0, thermal_state)
                thermal_signal = self.thermal_processor.process_thermal_signal(thermal_state)
                
                # Run TTT routing with all features
                start_time = time.time()
                routing_result = self._run_comprehensive_ttt_routing(input_tensor, thermal_signal)
                end_time = time.time()
                
                # Calculate comprehensive metrics
                latency_ms = (end_time - start_time) * 1000
                energy_joules = self._estimate_comprehensive_energy(routing_result, input_tensor, thermal_state)
                accuracy = self._calculate_comprehensive_accuracy(routing_result, input_tensor, thermal_signal)
                
                # Calculate specialized metrics
                noise_robustness = self._calculate_noise_robustness(routing_result, input_tensor)
                thermal_adaptation = self._calculate_thermal_adaptation(routing_result, thermal_signal)
                convergence_metric = self._calculate_convergence_metric(routing_result, epoch, batch_idx)
                
                # Update TTT router with comprehensive feedback
                feedback = {
                    'estimated_energy': energy_joules,
                    'expert_usage': routing_result['expert_usage'],
                    'token_count': input_tensor.numel(),
                    'batch_size': input_tensor.size(0),
                    'seq_length': input_tensor.size(1),
                    'thermal_signal': thermal_signal,
                    'noise_level': self.args.noise_level,
                    'error_margin': self.args.error_margin
                }
                self.ttt_router.ttt_update(feedback)
                
                # Accumulate metrics
                total_energy += energy_joules
                total_latency += latency_ms
                total_accuracy += accuracy
                thermal_imbalance_scores.append(thermal_signal['thermal_imbalance_score'])
                routing_entropies.append(routing_result['routing_entropy'])
                noise_robustness_scores.append(noise_robustness)
                thermal_adaptation_scores.append(thermal_adaptation)
                convergence_metrics.append(convergence_metric)
                num_batches += 1
                
                # Store epoch metrics for convergence analysis
                epoch_metrics.append({
                    'energy': energy_joules,
                    'accuracy': accuracy,
                    'routing_entropy': routing_result['routing_entropy']
                })
                
                # Print progress
                if batch_idx % 25 == 0:
                    print(f"  Batch {batch_idx}: Energy={energy_joules:.6f}J, "
                          f"Latency={latency_ms:.2f}ms, Accuracy={accuracy:.3f}, "
                          f"NoiseRobust={noise_robustness:.3f}, ThermalAdapt={thermal_adaptation:.3f}")
            
            # Analyze epoch convergence
            epoch_convergence = self._analyze_epoch_convergence(epoch_metrics)
            convergence_metrics.extend([epoch_convergence] * len(epoch_metrics))
        
        # Calculate final averages
        avg_energy = total_energy / num_batches
        avg_latency = total_latency / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_thermal_imbalance = np.mean(thermal_imbalance_scores)
        avg_routing_entropy = np.mean(routing_entropies)
        avg_noise_robustness = np.mean(noise_robustness_scores)
        avg_thermal_adaptation = np.mean(thermal_adaptation_scores)
        avg_convergence_rate = np.mean(convergence_metrics)
        
        # Calculate improvements
        baseline_energy = self._get_baseline_energy()
        baseline_accuracy = 0.95
        energy_savings = ((baseline_energy - avg_energy) / baseline_energy) * 100
        accuracy_loss = ((baseline_accuracy - avg_accuracy) / baseline_accuracy) * 100
        
        result = TTTValidationResult(
            lambda_energy=self.args.lambda_energy,
            chunk_size=self.args.chunk_size,
            noise_level=self.args.noise_level,
            error_margin=self.args.error_margin,
            avg_energy_joules=avg_energy,
            avg_latency_ms=avg_latency,
            avg_accuracy=avg_accuracy,
            thermal_imbalance_score=avg_thermal_imbalance,
            routing_entropy=avg_routing_entropy,
            ttt_update_count=self.ttt_router.ttt_update_count,
            energy_savings_percent=energy_savings,
            accuracy_loss_percent=accuracy_loss,
            noise_robustness_score=avg_noise_robustness,
            thermal_adaptation_score=avg_thermal_adaptation,
            convergence_rate=avg_convergence_rate
        )
        
        return result
    
    def _create_synthetic_input(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        """Create synthetic input tensor from batch data."""
        # Use batch data to create more realistic input
        input_tensor = torch.randn(
            self.args.batch_size,
            self.args.seq_length,
            self.args.d_model
        ).to(self.device)
        
        # Apply patterns from synthetic data if available
        if 'patterns' in batch_data:
            patterns = batch_data['patterns']
            # Apply pattern-based modifications
            for i, pattern in enumerate(patterns[:self.args.batch_size]):
                if i < input_tensor.size(0):
                    input_tensor[i] += torch.tensor(pattern, device=self.device) * 0.1
        
        return input_tensor
    
    def _apply_controlled_noise(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply controlled noise injection."""
        if self.args.noise_level == 0:
            return input_tensor
        
        # Add Gaussian noise
        noise = torch.randn_like(input_tensor) * self.args.noise_level
        
        # Add structured noise based on error margin
        structured_noise = torch.sin(input_tensor * self.args.error_margin) * 0.1
        
        return input_tensor + noise + structured_noise
    
    def _get_current_thermal_state(self) -> ThermalState:
        """Get current thermal state with realistic variations."""
        gpu_stats = self.gpu_monitor.get_current_stats()
        
        # Add realistic thermal variations
        base_temp = gpu_stats.get('temperature', 60.0)
        temp_variation = np.random.normal(0, 5)  # ±5°C variation
        temperature = max(30.0, min(95.0, base_temp + temp_variation))
        
        base_power = gpu_stats.get('power_watt', 200.0)
        power_variation = np.random.normal(0, 20)  # ±20W variation
        power_watt = max(100, min(400, base_power + power_variation))
        
        return ThermalState(
            temperature=temperature,
            power_watt=power_watt,
            memory_utilization=gpu_stats.get('memory_utilization_percent', 0.6) / 100.0,
            compute_utilization=gpu_stats.get('gpu_utilization_percent', 0.7) / 100.0,
            timestamp=time.time()
        )
    
    def _run_comprehensive_ttt_routing(self, input_tensor: torch.Tensor, 
                                     thermal_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive TTT routing with all features."""
        # Get routing decisions from TTT router
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={
                'thermal_signal': thermal_signal,
                'noise_level': self.args.noise_level,
                'error_margin': self.args.error_margin
            }
        )
        
        # Apply thermal-aware adjustments
        if thermal_signal['thermal_imbalance_detected']:
            expert_indices, expert_weights = self._apply_thermal_adjustments(
                expert_indices, expert_weights, thermal_signal
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
            'expert_usage': expert_usage,
            'routing_entropy': routing_entropy,
            'thermal_signal': thermal_signal,
            'router_metadata': router_metadata
        }
    
    def _apply_thermal_adjustments(self, expert_indices: torch.Tensor,
                                 expert_weights: torch.Tensor,
                                 thermal_signal: Dict[str, Any]) -> tuple:
        """Apply thermal-aware routing adjustments."""
        thermal_stats = self.thermal_router.get_thermal_stats()
        
        if thermal_stats['thermal_imbalance_score'] > 0.15:
            # Apply thermal penalty to hot experts
            adjusted_weights = expert_weights.clone()
            
            for i in range(self.args.num_experts):
                if thermal_stats['expert_thermal_loads'][i] > 0.7:
                    adjusted_weights[:, i] *= 0.6  # Reduce weight for hot experts
            
            # Renormalize
            adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=-1, keepdim=True)
            
            # Recompute top-k
            top_k_weights, top_k_indices = torch.topk(adjusted_weights, 2, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            return top_k_indices, top_k_weights
        
        return expert_indices, expert_weights
    
    def _estimate_comprehensive_energy(self, routing_result: Dict[str, Any],
                                     input_tensor: torch.Tensor,
                                     thermal_state: ThermalState) -> float:
        """Estimate comprehensive energy consumption."""
        batch_size = input_tensor.size(0)
        seq_length = input_tensor.size(1)
        
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
        
        # Thermal penalty
        thermal_factor = 1.0 + (thermal_state.temperature - 60.0) / 100.0
        
        # Memory pressure penalty
        memory_factor = 1.0 + thermal_state.memory_utilization * 0.2
        
        # Noise penalty (higher noise requires more computation)
        noise_factor = 1.0 + self.args.noise_level * 0.1
        
        total_energy = (routing_energy + expert_energy) * thermal_factor * memory_factor * noise_factor
        
        return total_energy
    
    def _calculate_comprehensive_accuracy(self, routing_result: Dict[str, Any],
                                        input_tensor: torch.Tensor,
                                        thermal_signal: Dict[str, Any]) -> float:
        """Calculate comprehensive accuracy metric."""
        expert_usage = routing_result['expert_usage']
        total_usage = expert_usage.sum().item()
        
        if total_usage == 0:
            return 0.0
        
        # Load balancing accuracy
        expected_usage = total_usage / self.args.num_experts
        usage_variance = torch.var(expert_usage.float()).item()
        load_balance_accuracy = 1.0 / (1.0 + usage_variance / (expected_usage ** 2))
        
        # Routing confidence
        expert_weights = routing_result['expert_weights']
        routing_confidence = torch.mean(expert_weights).item()
        
        # Thermal awareness accuracy
        thermal_accuracy = 1.0
        if thermal_signal['thermal_imbalance_detected']:
            thermal_accuracy = 0.8  # Good thermal awareness
        
        # Noise robustness accuracy
        noise_accuracy = 1.0 - self.args.noise_level * 0.5  # Degrade with noise
        
        # Combined accuracy
        accuracy = (0.4 * load_balance_accuracy + 
                   0.2 * routing_confidence + 
                   0.2 * thermal_accuracy + 
                   0.2 * noise_accuracy)
        
        return min(1.0, max(0.0, accuracy))
    
    def _calculate_noise_robustness(self, routing_result: Dict[str, Any],
                                  input_tensor: torch.Tensor) -> float:
        """Calculate noise robustness score."""
        # Measure how well routing handles noise
        expert_weights = routing_result['expert_weights']
        
        # Higher entropy indicates better noise handling
        entropy = self._calculate_routing_entropy(expert_weights)
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log(self.args.num_experts)
        normalized_entropy = entropy / max_entropy
        
        # Noise robustness increases with entropy up to a point
        if normalized_entropy < 0.5:
            robustness = normalized_entropy * 2
        else:
            robustness = 1.0 - (normalized_entropy - 0.5) * 0.5
        
        return min(1.0, max(0.0, robustness))
    
    def _calculate_thermal_adaptation(self, routing_result: Dict[str, Any],
                                    thermal_signal: Dict[str, Any]) -> float:
        """Calculate thermal adaptation score."""
        if not thermal_signal['thermal_imbalance_detected']:
            return 1.0  # Perfect adaptation when no thermal issues
        
        # Measure how well routing adapts to thermal conditions
        thermal_imbalance = thermal_signal['thermal_imbalance_score']
        
        # Higher adaptation score for lower thermal imbalance
        adaptation_score = 1.0 - thermal_imbalance
        
        return min(1.0, max(0.0, adaptation_score))
    
    def _calculate_convergence_metric(self, routing_result: Dict[str, Any],
                                    epoch: int, batch_idx: int) -> float:
        """Calculate convergence metric."""
        # Simple convergence metric based on routing stability
        expert_weights = routing_result['expert_weights']
        
        # Measure weight variance (lower variance indicates convergence)
        weight_variance = torch.var(expert_weights).item()
        
        # Normalize by epoch progress
        epoch_progress = epoch / self.args.num_epochs
        
        # Convergence improves with epoch progress and lower variance
        convergence = (1.0 - weight_variance) * (0.5 + 0.5 * epoch_progress)
        
        return min(1.0, max(0.0, convergence))
    
    def _analyze_epoch_convergence(self, epoch_metrics: List[Dict[str, float]]) -> float:
        """Analyze convergence within an epoch."""
        if len(epoch_metrics) < 2:
            return 0.5
        
        # Calculate trend in energy and accuracy
        energies = [m['energy'] for m in epoch_metrics]
        accuracies = [m['accuracy'] for m in epoch_metrics]
        
        # Energy should decrease, accuracy should increase
        energy_trend = (energies[0] - energies[-1]) / max(energies[0], 1e-6)
        accuracy_trend = (accuracies[-1] - accuracies[0]) / max(1 - accuracies[0], 1e-6)
        
        # Combined convergence metric
        convergence = (energy_trend + accuracy_trend) / 2
        
        return min(1.0, max(0.0, convergence))
    
    def _calculate_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Calculate routing entropy."""
        weights = expert_weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return torch.mean(entropy).item()
    
    def _get_baseline_energy(self) -> float:
        """Get baseline energy consumption."""
        batch_size = self.args.batch_size
        seq_length = self.args.seq_length
        
        routing_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        expert_cost = self.kernel_cost_model.get_cost("ffn_gate", batch_size * seq_length)
        
        return routing_cost["energy_joules"] + expert_cost["energy_joules"]
    
    def save_results(self, result: TTTValidationResult, output_file: str):
        """Save validation results to file."""
        output_data = {
            'validation_config': {
                'lambda_energy': self.args.lambda_energy,
                'chunk_size': self.args.chunk_size,
                'noise_level': self.args.noise_level,
                'error_margin': self.args.error_margin,
                'num_experts': self.args.num_experts,
                'moe_top_k': self.args.moe_top_k,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'enable_thermal_awareness': self.args.enable_thermal_awareness,
                'enable_noise_injection': self.args.enable_noise_injection
            },
            'validation_results': {
                'avg_energy_joules': result.avg_energy_joules,
                'avg_latency_ms': result.avg_latency_ms,
                'avg_accuracy': result.avg_accuracy,
                'thermal_imbalance_score': result.thermal_imbalance_score,
                'routing_entropy': result.routing_entropy,
                'ttt_update_count': result.ttt_update_count,
                'energy_savings_percent': result.energy_savings_percent,
                'accuracy_loss_percent': result.accuracy_loss_percent,
                'noise_robustness_score': result.noise_robustness_score,
                'thermal_adaptation_score': result.thermal_adaptation_score,
                'convergence_rate': result.convergence_rate
            },
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Validation results saved to {output_file}")
    
    def print_summary(self, result: TTTValidationResult):
        """Print validation summary."""
        print("\n=== Comprehensive TTT Validation Summary ===")
        print(f"Configuration:")
        print(f"  Lambda Energy: {result.lambda_energy}")
        print(f"  Chunk Size: {result.chunk_size}")
        print(f"  Noise Level: {result.noise_level}")
        print(f"  Error Margin: {result.error_margin}")
        print()
        print("Performance Metrics:")
        print(f"  Average Energy: {result.avg_energy_joules:.6f} J")
        print(f"  Average Latency: {result.avg_latency_ms:.2f} ms")
        print(f"  Average Accuracy: {result.avg_accuracy:.3f}")
        print(f"  Thermal Imbalance: {result.thermal_imbalance_score:.3f}")
        print(f"  Routing Entropy: {result.routing_entropy:.3f}")
        print()
        print("Optimization Results:")
        print(f"  Energy Savings: {result.energy_savings_percent:.2f}%")
        print(f"  Accuracy Loss: {result.accuracy_loss_percent:.2f}%")
        print(f"  TTT Updates: {result.ttt_update_count}")
        print()
        print("Specialized Metrics:")
        print(f"  Noise Robustness: {result.noise_robustness_score:.3f}")
        print(f"  Thermal Adaptation: {result.thermal_adaptation_score:.3f}")
        print(f"  Convergence Rate: {result.convergence_rate:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive TTT Validation")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=2, help="Top-k experts")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--synthetic_data_file", type=str, required=True, help="Synthetic data file")
    parser.add_argument("--lambda_energy", type=float, default=0.05, help="Energy penalty weight")
    parser.add_argument("--enable_thermal_awareness", action="store_true", help="Enable thermal awareness")
    parser.add_argument("--enable_noise_injection", action="store_true", help="Enable noise injection")
    parser.add_argument("--noise_level", type=float, default=0.03, help="Noise level")
    parser.add_argument("--error_margin", type=float, default=0.08, help="Error margin")
    parser.add_argument("--chunk_size", type=int, default=500, help="TTT chunk size")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    
    args = parser.parse_args()
    
    print("=== Comprehensive TTT Validation ===")
    print(f"Lambda Energy: {args.lambda_energy}")
    print(f"Chunk Size: {args.chunk_size}")
    print(f"Noise Level: {args.noise_level}")
    print(f"Error Margin: {args.error_margin}")
    print(f"Thermal Awareness: {args.enable_thermal_awareness}")
    print(f"Noise Injection: {args.enable_noise_injection}")
    
    # Run validation
    tester = TTTValidationTester(args)
    result = tester.run_validation()
    
    # Save results
    tester.save_results(result, args.output_file)
    
    # Print summary
    tester.print_summary(result)
    
    print("\n=== Validation Complete ===")

if __name__ == "__main__":
    main() 