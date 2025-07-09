#!/usr/bin/env python3
"""
Test script for error margin and noise analysis with synthetic data.
Tests system robustness to different noise levels and error margins.
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

from src.kernelcostmodel import KernelCostModel
from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class ErrorMarginTestResult:
    """Results from error margin analysis."""
    noise_level: float
    error_margin: float
    avg_energy_joules: float
    avg_accuracy: float
    robustness_score: float
    routing_stability: float
    convergence_rate: float

class ErrorMarginAnalyzer:
    """
    Analyze error margins and noise robustness with synthetic data.
    Tests system behavior under different noise conditions.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        
        # TTT Router
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=2,
            lambda_energy=0.05
        ).to(self.device)
        
        # Load synthetic data
        self.synthetic_data = self._load_synthetic_data(args.synthetic_data_file)
        
        # Parse noise levels and error margins
        self.noise_levels = [float(x) for x in args.noise_levels.split(',')]
        self.error_margins = [float(x) for x in args.error_margins.split(',')]
        
    def _load_synthetic_data(self, data_file: str) -> List[Dict[str, Any]]:
        """Load synthetic dataset."""
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data['data_points']
    
    def run_analysis(self) -> List[ErrorMarginTestResult]:
        """Run comprehensive error margin analysis."""
        print("Running Error Margin and Noise Analysis")
        print(f"Noise Levels: {self.noise_levels}")
        print(f"Error Margins: {self.error_margins}")
        
        results = []
        
        for noise_level in self.noise_levels:
            for error_margin in self.error_margins:
                print(f"\nTesting noise_level={noise_level}, error_margin={error_margin}")
                
                result = self._test_noise_error_combination(noise_level, error_margin)
                results.append(result)
                
                print(f"  Energy: {result.avg_energy_joules:.6f} J")
                print(f"  Accuracy: {result.avg_accuracy:.3f}")
                print(f"  Robustness: {result.robustness_score:.3f}")
                print(f"  Stability: {result.routing_stability:.3f}")
                print(f"  Convergence: {result.convergence_rate:.3f}")
        
        return results
    
    def _test_noise_error_combination(self, noise_level: float, error_margin: float) -> ErrorMarginTestResult:
        """Test a specific noise level and error margin combination."""
        # Initialize metrics
        total_energy = 0.0
        total_accuracy = 0.0
        robustness_scores = []
        routing_stabilities = []
        convergence_rates = []
        
        num_batches = 0
        
        for batch_idx in range(self.args.num_batches):
            # Create synthetic input
            input_tensor = torch.randn(
                self.args.batch_size,
                self.args.seq_length,
                self.args.d_model
            ).to(self.device)
            
            # Apply noise and error margin
            noisy_tensor = self._apply_noise_and_error(input_tensor, noise_level, error_margin)
            
            # Run routing
            start_time = time.time()
            routing_result = self._run_noise_robust_routing(noisy_tensor, noise_level, error_margin)
            end_time = time.time()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            energy_joules = self._estimate_noise_energy(routing_result, noisy_tensor, noise_level)
            accuracy = self._calculate_noise_accuracy(routing_result, noisy_tensor, input_tensor)
            
            # Calculate specialized metrics
            robustness = self._calculate_robustness_score(routing_result, noise_level, error_margin)
            stability = self._calculate_routing_stability(routing_result)
            convergence = self._calculate_convergence_rate(routing_result, batch_idx)
            
            # Update TTT router
            feedback = {
                'estimated_energy': energy_joules,
                'expert_usage': routing_result['expert_usage'],
                'token_count': noisy_tensor.numel(),
                'batch_size': noisy_tensor.size(0),
                'seq_length': noisy_tensor.size(1),
                'noise_level': noise_level,
                'error_margin': error_margin
            }
            self.ttt_router.ttt_update(feedback)
            
            # Accumulate metrics
            total_energy += energy_joules
            total_accuracy += accuracy
            robustness_scores.append(robustness)
            routing_stabilities.append(stability)
            convergence_rates.append(convergence)
            num_batches += 1
        
        # Calculate averages
        avg_energy = total_energy / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_robustness = np.mean(robustness_scores)
        avg_stability = np.mean(routing_stabilities)
        avg_convergence = np.mean(convergence_rates)
        
        return ErrorMarginTestResult(
            noise_level=noise_level,
            error_margin=error_margin,
            avg_energy_joules=avg_energy,
            avg_accuracy=avg_accuracy,
            robustness_score=avg_robustness,
            routing_stability=avg_stability,
            convergence_rate=avg_convergence
        )
    
    def _apply_noise_and_error(self, input_tensor: torch.Tensor, 
                              noise_level: float, error_margin: float) -> torch.Tensor:
        """Apply noise and error margin to input tensor."""
        # Add Gaussian noise
        if noise_level > 0:
            noise = torch.randn_like(input_tensor) * noise_level
            input_tensor = input_tensor + noise
        
        # Add error margin effects
        if error_margin > 0:
            # Simulate quantization-like errors
            scale_factor = 1.0 + error_margin
            input_tensor = torch.round(input_tensor * scale_factor) / scale_factor
            
            # Add systematic bias
            bias = torch.sin(input_tensor * error_margin) * 0.1
            input_tensor = input_tensor + bias
        
        return input_tensor
    
    def _run_noise_robust_routing(self, input_tensor: torch.Tensor,
                                noise_level: float, error_margin: float) -> Dict[str, Any]:
        """Run routing with noise robustness."""
        # Get routing decisions from TTT router
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={
                'noise_level': noise_level,
                'error_margin': error_margin
            }
        )
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.args.num_experts, device=self.device)
        for i in range(self.args.num_experts):
            expert_usage[i] = (expert_indices == i).sum().item()
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'expert_usage': expert_usage,
            'router_metadata': router_metadata,
            'noise_level': noise_level,
            'error_margin': error_margin
        }
    
    def _estimate_noise_energy(self, routing_result: Dict[str, Any],
                             input_tensor: torch.Tensor, noise_level: float) -> float:
        """Estimate energy consumption considering noise."""
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
        
        # Noise penalty (higher noise requires more computation)
        noise_factor = 1.0 + noise_level * 0.2
        
        total_energy = (routing_energy + expert_energy) * noise_factor
        
        return total_energy
    
    def _calculate_noise_accuracy(self, routing_result: Dict[str, Any],
                                noisy_tensor: torch.Tensor,
                                original_tensor: torch.Tensor) -> float:
        """Calculate accuracy considering noise effects."""
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
        
        # Noise robustness accuracy (degrade with noise)
        noise_level = routing_result['noise_level']
        noise_robustness = 1.0 - noise_level * 0.5
        
        # Error margin robustness
        error_margin = routing_result['error_margin']
        error_robustness = 1.0 - error_margin * 0.3
        
        # Combined accuracy
        accuracy = (0.4 * load_balance_accuracy + 
                   0.2 * routing_confidence + 
                   0.2 * noise_robustness + 
                   0.2 * error_robustness)
        
        return min(1.0, max(0.0, accuracy))
    
    def _calculate_robustness_score(self, routing_result: Dict[str, Any],
                                  noise_level: float, error_margin: float) -> float:
        """Calculate robustness score."""
        # Measure how well routing handles noise and errors
        expert_weights = routing_result['expert_weights']
        
        # Calculate routing entropy
        weights = expert_weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        avg_entropy = torch.mean(entropy).item()
        
        # Normalize entropy to [0, 1] range
        max_entropy = np.log(self.args.num_experts)
        normalized_entropy = avg_entropy / max_entropy
        
        # Robustness increases with entropy up to a point
        if normalized_entropy < 0.6:
            robustness = normalized_entropy / 0.6
        else:
            robustness = 1.0 - (normalized_entropy - 0.6) * 0.5
        
        # Penalize for high noise/error levels
        noise_penalty = noise_level * 0.3
        error_penalty = error_margin * 0.2
        
        robustness = robustness * (1.0 - noise_penalty - error_penalty)
        
        return min(1.0, max(0.0, robustness))
    
    def _calculate_routing_stability(self, routing_result: Dict[str, Any]) -> float:
        """Calculate routing stability score."""
        expert_weights = routing_result['expert_weights']
        
        # Measure weight variance (lower variance = higher stability)
        weight_variance = torch.var(expert_weights).item()
        
        # Convert to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + weight_variance * 10)
        
        return min(1.0, max(0.0, stability))
    
    def _calculate_convergence_rate(self, routing_result: Dict[str, Any], batch_idx: int) -> float:
        """Calculate convergence rate."""
        # Simple convergence metric based on TTT update count
        ttt_update_count = routing_result['router_metadata'].get('ttt_update_count', 0)
        
        # Convergence improves with more updates up to a point
        if ttt_update_count < 100:
            convergence = ttt_update_count / 100
        else:
            convergence = 1.0 - (ttt_update_count - 100) / 1000
        
        # Also consider batch progress
        batch_progress = batch_idx / self.args.num_batches
        convergence = convergence * (0.5 + 0.5 * batch_progress)
        
        return min(1.0, max(0.0, convergence))
    
    def save_results(self, results: List[ErrorMarginTestResult], output_file: str):
        """Save analysis results to file."""
        output_data = {
            'analysis_config': {
                'num_experts': self.args.num_experts,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'num_batches': self.args.num_batches,
                'noise_levels': self.noise_levels,
                'error_margins': self.error_margins
            },
            'results': [
                {
                    'noise_level': r.noise_level,
                    'error_margin': r.error_margin,
                    'avg_energy_joules': r.avg_energy_joules,
                    'avg_accuracy': r.avg_accuracy,
                    'robustness_score': r.robustness_score,
                    'routing_stability': r.routing_stability,
                    'convergence_rate': r.convergence_rate
                }
                for r in results
            ],
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[ErrorMarginTestResult]):
        """Print analysis summary."""
        print("\n=== Error Margin Analysis Summary ===")
        
        # Group by noise level
        noise_groups = {}
        for result in results:
            if result.noise_level not in noise_groups:
                noise_groups[result.noise_level] = []
            noise_groups[result.noise_level].append(result)
        
        for noise_level, group_results in noise_groups.items():
            print(f"\nNoise Level: {noise_level}")
            for result in group_results:
                print(f"  Error Margin {result.error_margin}: "
                      f"Energy={result.avg_energy_joules:.6f}J, "
                      f"Accuracy={result.avg_accuracy:.3f}, "
                      f"Robustness={result.robustness_score:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Error Margins and Noise")
    parser.add_argument("--synthetic_data_file", type=str, required=True, help="Synthetic data file")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--noise_levels", type=str, default="0.0,0.01,0.05,0.1,0.2",
                       help="Comma-separated noise levels")
    parser.add_argument("--error_margins", type=str, default="0.05,0.1,0.15,0.2",
                       help="Comma-separated error margins")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    
    args = parser.parse_args()
    
    print("=== Error Margin and Noise Analysis ===")
    print(f"Noise Levels: {args.noise_levels}")
    print(f"Error Margins: {args.error_margins}")
    
    # Run analysis
    analyzer = ErrorMarginAnalyzer(args)
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results(results, args.output_file)
    
    # Print summary
    analyzer.print_summary(results)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main() 