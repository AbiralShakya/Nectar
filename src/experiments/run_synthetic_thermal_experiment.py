#!/usr/bin/env python3
"""
Synthetic Thermal-Aware Energy-Adaptive Routing Experiment
Tests TTT routing with synthetic thermal stress and energy optimization.
Tracks expert diversity and temperature-based adaptation over time.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import argparse
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.ttt_router import EnergyAwareTTTRouter
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState

@dataclass
class ExpertUsage:
    """Track expert usage and thermal state."""
    expert_id: int
    usage_count: int
    temperature: float
    energy_cost: float
    timestamp: float

@dataclass
class TimeStepData:
    """Data for each time step."""
    step: int
    timestamp: float
    expert_usage: List[ExpertUsage]
    total_energy: float
    thermal_imbalance: float
    routing_entropy: float
    expert_diversity: float
    avg_temperature: float

class SyntheticThermalExperiment:
    """
    Synthetic experiment for thermal-aware energy-adaptive routing.
    Creates 8 experts with synthetic thermal stress (6 cold, 2 hot).
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize router
        self.router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=args.moe_top_k,
            lambda_energy=args.lambda_energy
        ).to(self.device)
        
        # Initialize thermal router
        self.thermal_router = ThermalAwareRouter(
            num_experts=args.num_experts,
            num_gpus=1
        )
        
        # Initialize GPU monitor
        self.gpu_monitor = GpuSystemMonitor()
        
        # Synthetic thermal configuration
        self.cold_experts = [0, 1, 2, 3, 4, 5]  # 6 cold experts (0°C)
        self.hot_experts = [6, 7]               # 2 hot experts (normal heat)
        
        # Track experiment data
        self.time_series_data: List[TimeStepData] = []
        self.expert_usage_history: Dict[int, List[int]] = {i: [] for i in range(args.num_experts)}
        
        # Generate synthetic data
        self.synthetic_data = self._generate_synthetic_data()
        
        print(f"Initialized experiment with {args.num_experts} experts")
        print(f"Cold experts (0°C): {self.cold_experts}")
        print(f"Hot experts (normal): {self.hot_experts}")
    
    def _generate_synthetic_data(self) -> torch.Tensor:
        """Generate synthetic input data."""
        print("Generating synthetic data...")
        
        # Create realistic patterns for MoE routing
        data = torch.randn(
            self.args.num_batches * self.args.batch_size,
            self.args.seq_length,
            self.args.d_model
        )
        
        # Add some structure to make routing more interesting
        for i in range(data.size(0)):
            # Add bias to different dimensions to create routing patterns
            data[i, :, :self.args.d_model//4] += torch.randn(1) * 0.5
            data[i, :, self.args.d_model//4:self.args.d_model//2] += torch.randn(1) * 0.3
        
        print(f"Generated {data.size(0)} synthetic data points")
        return data
    
    def _get_synthetic_thermal_state(self, expert_id: int) -> float:
        """Get synthetic temperature for an expert."""
        if expert_id in self.cold_experts:
            return 0.0  # Cold experts at 0°C
        else:
            # Hot experts at normal temperature (60-80°C)
            return 60.0 + np.random.normal(0, 5)
    
    def _calculate_expert_energy_cost(self, expert_id: int, usage_count: int) -> float:
        """Calculate energy cost for an expert based on its thermal state."""
        base_energy = 1.0  # Base energy per token
        
        if expert_id in self.cold_experts:
            # Cold experts are energy-efficient
            return base_energy * usage_count * 0.5
        else:
            # Hot experts are energy-inefficient
            return base_energy * usage_count * 2.0
    
    def _calculate_expert_diversity(self, expert_usage: List[int]) -> float:
        """Calculate expert diversity (entropy of usage distribution)."""
        if sum(expert_usage) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = np.array(expert_usage) / sum(expert_usage)
        probs = probs[probs > 0]  # Remove zeros
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def _calculate_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Calculate routing entropy."""
        weights = expert_weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return torch.mean(entropy).item()
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the synthetic thermal experiment."""
        print(f"Running synthetic thermal experiment for {self.args.num_batches} batches...")
        
        total_energy = 0.0
        step = 0
        
        for batch_idx in range(self.args.num_batches):
            # Get batch data
            start_idx = batch_idx * self.args.batch_size
            end_idx = start_idx + self.args.batch_size
            batch_data = self.synthetic_data[start_idx:end_idx].to(self.device)
            
            # Run routing
            expert_indices, expert_weights, router_metadata = self.router(batch_data)
            
            # Calculate expert usage
            expert_usage = torch.zeros(self.args.num_experts, device=self.device)
            for i in range(self.args.num_experts):
                expert_usage[i] = (expert_indices == i).sum().item()
            
            # Get thermal states for each expert
            expert_thermal_states = []
            batch_energy = 0.0
            
            for expert_id in range(self.args.num_experts):
                temperature = self._get_synthetic_thermal_state(expert_id)
                usage_count = int(expert_usage[expert_id].item())
                energy_cost = self._calculate_expert_energy_cost(expert_id, usage_count)
                
                expert_thermal_states.append(ExpertUsage(
                    expert_id=expert_id,
                    usage_count=usage_count,
                    temperature=temperature,
                    energy_cost=energy_cost,
                    timestamp=time.time()
                ))
                
                batch_energy += energy_cost
                self.expert_usage_history[expert_id].append(usage_count)
            
            total_energy += batch_energy
            
            # Calculate metrics
            routing_entropy = self._calculate_routing_entropy(expert_weights)
            expert_diversity = self._calculate_expert_diversity(expert_usage.cpu().numpy())
            
            # Calculate thermal imbalance
            cold_usage = sum(expert_usage[i].item() for i in self.cold_experts)
            hot_usage = sum(expert_usage[i].item() for i in self.hot_experts)
            total_usage = expert_usage.sum().item()
            
            if total_usage > 0:
                thermal_imbalance = abs(cold_usage - hot_usage) / total_usage
            else:
                thermal_imbalance = 0.0
            
            # Calculate average temperature
            avg_temperature = np.mean([exp.temperature for exp in expert_thermal_states])
            
            # Store time step data
            time_step = TimeStepData(
                step=step,
                timestamp=time.time(),
                expert_usage=expert_thermal_states,
                total_energy=batch_energy,
                thermal_imbalance=thermal_imbalance,
                routing_entropy=routing_entropy,
                expert_diversity=expert_diversity,
                avg_temperature=avg_temperature
            )
            self.time_series_data.append(time_step)
            
            # TTT update
            feedback = {
                'estimated_energy': batch_energy,
                'expert_usage': expert_usage,
                'token_count': batch_data.numel(),
                'batch_size': batch_data.size(0),
                'seq_length': batch_data.size(1)
            }
            self.router.ttt_update(feedback)
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}: Energy={batch_energy:.2f}J, "
                      f"Diversity={expert_diversity:.3f}, "
                      f"Thermal Imbalance={thermal_imbalance:.3f}, "
                      f"Cold/Hot={cold_usage}/{hot_usage}")
            
            step += 1
        
        # Compile results
        results = self._compile_results()
        return results
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile experiment results."""
        print("Compiling results...")
        
        # Calculate final metrics
        total_steps = len(self.time_series_data)
        total_energy = sum(step.total_energy for step in self.time_series_data)
        avg_diversity = np.mean([step.expert_diversity for step in self.time_series_data])
        avg_thermal_imbalance = np.mean([step.thermal_imbalance for step in self.time_series_data])
        
        # Calculate expert usage statistics
        final_expert_usage = {}
        for expert_id in range(self.args.num_experts):
            usage_history = self.expert_usage_history[expert_id]
            final_expert_usage[expert_id] = {
                'total_usage': sum(usage_history),
                'avg_usage': np.mean(usage_history) if usage_history else 0,
                'usage_trend': usage_history,
                'is_cold': expert_id in self.cold_experts,
                'is_hot': expert_id in self.hot_experts
            }
        
        # Calculate energy savings (compared to baseline)
        baseline_energy = total_steps * self.args.batch_size * self.args.seq_length * 1.5  # Assume baseline
        energy_savings = ((baseline_energy - total_energy) / baseline_energy) * 100
        
        results = {
            'experiment_config': {
                'num_experts': self.args.num_experts,
                'cold_experts': self.cold_experts,
                'hot_experts': self.hot_experts,
                'lambda_energy': self.args.lambda_energy,
                'num_batches': self.args.num_batches,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model
            },
            'results': {
                'total_energy': float(total_energy),
                'avg_diversity': float(avg_diversity),
                'avg_thermal_imbalance': float(avg_thermal_imbalance),
                'energy_savings_percent': float(energy_savings),
                'total_steps': total_steps,
                'ttt_updates': self.router.ttt_update_count
            },
            'expert_usage': final_expert_usage,
            'time_series_data': [
                {
                    'step': step.step,
                    'timestamp': step.timestamp,
                    'total_energy': step.total_energy,
                    'thermal_imbalance': step.thermal_imbalance,
                    'routing_entropy': step.routing_entropy,
                    'expert_diversity': step.expert_diversity,
                    'avg_temperature': step.avg_temperature,
                    'expert_usage': [
                        {
                            'expert_id': exp.expert_id,
                            'usage_count': exp.usage_count,
                            'temperature': exp.temperature,
                            'energy_cost': exp.energy_cost
                        }
                        for exp in step.expert_usage
                    ]
                }
                for step in self.time_series_data
            ],
            'timestamp': time.time()
        }
        
        return results
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Create comprehensive visualizations."""
        print("Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Expert Usage Over Time
        self._plot_expert_usage_over_time(results, output_path)
        
        # 2. Temperature vs Usage Scatter
        self._plot_temperature_vs_usage(results, output_path)
        
        # 3. Expert Diversity Growth
        self._plot_diversity_growth(results, output_path)
        
        # 4. Energy vs Thermal Imbalance
        self._plot_energy_vs_thermal(results, output_path)
        
        # 5. Cold vs Hot Expert Comparison
        self._plot_cold_vs_hot_comparison(results, output_path)
        
        print(f"Visualizations saved to {output_path}")
    
    def _plot_expert_usage_over_time(self, results: Dict[str, Any], output_path: Path):
        """Plot expert usage over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        time_series = results['time_series_data']
        steps = [step['step'] for step in time_series]
        
        # Plot individual expert usage
        for expert_id in range(self.args.num_experts):
            usage_trend = results['expert_usage'][expert_id]['usage_trend']
            if expert_id in self.cold_experts:
                ax1.plot(steps, usage_trend, label=f'Cold Expert {expert_id}', alpha=0.7)
            else:
                ax2.plot(steps, usage_trend, label=f'Hot Expert {expert_id}', alpha=0.7)
        
        ax1.set_title('Cold Expert Usage Over Time')
        ax1.set_ylabel('Usage Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Hot Expert Usage Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Usage Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'expert_usage_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temperature_vs_usage(self, results: Dict[str, Any], output_path: Path):
        """Plot temperature vs usage scatter."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect data points
        temperatures = []
        usages = []
        colors = []
        
        for expert_id in range(self.args.num_experts):
            total_usage = results['expert_usage'][expert_id]['total_usage']
            avg_temp = np.mean([
                step['expert_usage'][expert_id]['temperature'] 
                for step in results['time_series_data']
            ])
            
            temperatures.append(avg_temp)
            usages.append(total_usage)
            colors.append('blue' if expert_id in self.cold_experts else 'red')
        
        # Create scatter plot
        scatter = ax.scatter(temperatures, usages, c=colors, s=100, alpha=0.7)
        
        # Add labels
        for i, expert_id in enumerate(range(self.args.num_experts)):
            ax.annotate(f'E{expert_id}', (temperatures[i], usages[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Average Temperature (°C)')
        ax.set_ylabel('Total Usage Count')
        ax.set_title('Expert Usage vs Temperature')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.7, label='Cold Experts (0°C)'),
            Patch(facecolor='red', alpha=0.7, label='Hot Experts (60-80°C)')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_path / 'temperature_vs_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_diversity_growth(self, results: Dict[str, Any], output_path: Path):
        """Plot expert diversity growth over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_series = results['time_series_data']
        steps = [step['step'] for step in time_series]
        diversity = [step['expert_diversity'] for step in time_series]
        
        ax.plot(steps, diversity, linewidth=2, color='green')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Expert Diversity (Entropy)')
        ax.set_title('Expert Diversity Growth Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(steps, diversity, 1)
        p = np.poly1d(z)
        ax.plot(steps, p(steps), "--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'diversity_growth.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_vs_thermal(self, results: Dict[str, Any], output_path: Path):
        """Plot energy vs thermal imbalance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        time_series = results['time_series_data']
        energy = [step['total_energy'] for step in time_series]
        thermal_imbalance = [step['thermal_imbalance'] for step in time_series]
        
        scatter = ax.scatter(thermal_imbalance, energy, c=range(len(energy)), 
                           cmap='viridis', alpha=0.7)
        
        ax.set_xlabel('Thermal Imbalance')
        ax.set_ylabel('Energy Consumption (J)')
        ax.set_title('Energy vs Thermal Imbalance')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time Step')
        
        plt.tight_layout()
        plt.savefig(output_path / 'energy_vs_thermal.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cold_vs_hot_comparison(self, results: Dict[str, Any], output_path: Path):
        """Plot cold vs hot expert comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Calculate statistics
        cold_usage = []
        hot_usage = []
        cold_energy = []
        hot_energy = []
        
        for step in results['time_series_data']:
            cold_step_usage = sum(
                exp['usage_count'] for exp in step['expert_usage'] 
                if exp['expert_id'] in self.cold_experts
            )
            hot_step_usage = sum(
                exp['usage_count'] for exp in step['expert_usage'] 
                if exp['expert_id'] in self.hot_experts
            )
            
            cold_step_energy = sum(
                exp['energy_cost'] for exp in step['expert_usage'] 
                if exp['expert_id'] in self.cold_experts
            )
            hot_step_energy = sum(
                exp['energy_cost'] for exp in step['expert_usage'] 
                if exp['expert_id'] in self.hot_experts
            )
            
            cold_usage.append(cold_step_usage)
            hot_usage.append(hot_step_usage)
            cold_energy.append(cold_step_energy)
            hot_energy.append(hot_step_energy)
        
        steps = list(range(len(cold_usage)))
        
        # Plot 1: Usage comparison
        ax1.plot(steps, cold_usage, label='Cold Experts', color='blue')
        ax1.plot(steps, hot_usage, label='Hot Experts', color='red')
        ax1.set_title('Usage Comparison')
        ax1.set_ylabel('Usage Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy comparison
        ax2.plot(steps, cold_energy, label='Cold Experts', color='blue')
        ax2.plot(steps, hot_energy, label='Hot Experts', color='red')
        ax2.set_title('Energy Consumption')
        ax2.set_ylabel('Energy (J)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Usage ratio over time
        total_usage = [c + h for c, h in zip(cold_usage, hot_usage)]
        cold_ratio = [c / t if t > 0 else 0 for c, t in zip(cold_usage, total_usage)]
        hot_ratio = [h / t if t > 0 else 0 for h, t in zip(hot_usage, total_usage)]
        
        ax3.plot(steps, cold_ratio, label='Cold Ratio', color='blue')
        ax3.plot(steps, hot_ratio, label='Hot Ratio', color='red')
        ax3.set_title('Usage Ratio Over Time')
        ax3.set_ylabel('Ratio')
        ax3.set_xlabel('Time Step')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Energy efficiency
        cold_efficiency = [c / (c + h) if (c + h) > 0 else 0 for c, h in zip(cold_usage, hot_usage)]
        ax4.plot(steps, cold_efficiency, color='green')
        ax4.set_title('Cold Expert Efficiency')
        ax4.set_ylabel('Efficiency Ratio')
        ax4.set_xlabel('Time Step')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cold_vs_hot_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Synthetic Thermal-Aware Energy-Adaptive Routing Experiment")
    parser.add_argument("--lambda_energy", type=float, default=0.1, help="Energy penalty weight")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=2, help="Top-k experts to route to")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--output_dir", type=str, default="results/thermal_experiment", help="Output directory")
    parser.add_argument("--output_file", type=str, default="thermal_experiment_results.json", help="Output file")
    
    args = parser.parse_args()
    
    print("=== Synthetic Thermal-Aware Energy-Adaptive Routing Experiment ===")
    print(f"Lambda Energy: {args.lambda_energy}")
    print(f"Number of Experts: {args.num_experts}")
    print(f"Cold Experts: 6 (0°C)")
    print(f"Hot Experts: 2 (60-80°C)")
    
    # Run experiment
    experiment = SyntheticThermalExperiment(args)
    results = experiment.run_experiment()
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations
    experiment.create_visualizations(results, args.output_dir)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    print(f"Total Energy: {results['results']['total_energy']:.2f}J")
    print(f"Average Diversity: {results['results']['avg_diversity']:.3f}")
    print(f"Average Thermal Imbalance: {results['results']['avg_thermal_imbalance']:.3f}")
    print(f"Energy Savings: {results['results']['energy_savings_percent']:.2f}%")
    print(f"TTT Updates: {results['results']['ttt_updates']}")
    
    print(f"\nResults saved to: {output_path / args.output_file}")
    print(f"Visualizations saved to: {output_path}")

if __name__ == "__main__":
    main() 