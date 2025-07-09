import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import time
from typing import Dict, Tuple, Any, List
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.moe_models import MoEConfig, MoETransformerBlock, OptimizedMoELayer, SwiGLUExpert, OptimizedQuantizedExpert, DistributedMoELayer, NetworkTopologyOptimizer
from routers import RoutingStrategy, GpuSystemMonitor, AdaptiveRouter
from src.kernelcostmodel import KernelCostModel
from data_utils import DataLoaderManager
from metrics_logger import MetricsLogger
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState
from models.ttt_router import LaCTEnergyAwareTTTRouter

@dataclass
class OptimizationTarget:
    """Defines optimization target and constraints."""
    target_type: str  # "power", "runtime", "balanced", "thermal"
    power_budget: float = 300.0  # Watts
    latency_budget: float = 50.0  # ms
    thermal_budget: float = 80.0  # °C
    accuracy_threshold: float = 0.95  # Minimum accuracy to maintain

@dataclass
class OptimizationResult:
    """Results from optimization experiment."""
    target_type: str
    batch_size: int
    expert_config: Dict[str, Any]
    achieved_power: float
    achieved_latency: float
    achieved_accuracy: float
    thermal_imbalance: float
    energy_efficiency: float  # tokens per joule
    throughput: float  # tokens per second

class DynamicAdaptationExperiment:
    """
    Comprehensive experiment to test power vs runtime optimization trade-offs.
    Addresses the key question: is optimizing for power the same as optimizing for performance?
    """
    def __init__(self, num_gpus: int = 8, num_experts: int = 16, d_model: int = 768):
        self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.d_model = d_model
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        self.gpu_monitor = GpuSystemMonitor()
        self.thermal_router = ThermalAwareRouter(num_experts, num_gpus)
        
        # Optimization targets
        self.optimization_targets = [
            OptimizationTarget("power", power_budget=250.0),
            OptimizationTarget("runtime", latency_budget=30.0),
            OptimizationTarget("balanced", power_budget=300.0, latency_budget=40.0),
            OptimizationTarget("thermal", thermal_budget=75.0)
        ]
        
        # Batch size configurations to test
        self.batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        
        # Expert configurations to test
        self.expert_configs = [
            {"top_k": 1, "expert_capacity": 0.5},
            {"top_k": 2, "expert_capacity": 0.7},
            {"top_k": 4, "expert_capacity": 0.9},
            {"top_k": 2, "expert_capacity": 1.0},
        ]
        
        # Results storage
        self.results = []
        self.detailed_logs = []
        
    def run_optimization_experiment(self) -> List[OptimizationResult]:
        """Run comprehensive optimization experiment."""
        print("Starting Dynamic Adaptation Experiment...")
        print(f"Testing {len(self.optimization_targets)} optimization targets")
        print(f"Testing {len(self.batch_sizes)} batch sizes")
        print(f"Testing {len(self.expert_configs)} expert configurations")
        
        for target in self.optimization_targets:
            print(f"\n--- Testing {target.target_type} optimization ---")
            
            for batch_size in self.batch_sizes:
                for expert_config in self.expert_configs:
                    result = self._test_configuration(target, batch_size, expert_config)
                    if result:
                        self.results.append(result)
                        self._log_detailed_result(result)
        
        return self.results
    
    def _test_configuration(self, target: OptimizationTarget, batch_size: int, 
                           expert_config: Dict[str, Any]) -> Optional[OptimizationResult]:
        """Test a specific configuration."""
        try:
            # Create synthetic workload
            input_tensor = torch.randn(batch_size, 512, self.d_model)
            
            # Measure baseline performance
            baseline_metrics = self._measure_baseline_performance(input_tensor, expert_config)
            
            # Apply optimization based on target
            optimized_metrics = self._apply_optimization(target, input_tensor, expert_config, baseline_metrics)
            
            # Calculate efficiency metrics
            energy_efficiency = batch_size * 512 / (optimized_metrics['energy_joules'] + 1e-8)
            throughput = batch_size * 512 / (optimized_metrics['latency_ms'] / 1000 + 1e-8)
            
            return OptimizationResult(
                target_type=target.target_type,
                batch_size=batch_size,
                expert_config=expert_config,
                achieved_power=optimized_metrics['power_watt'],
                achieved_latency=optimized_metrics['latency_ms'],
                achieved_accuracy=optimized_metrics['accuracy'],
                thermal_imbalance=optimized_metrics['thermal_imbalance'],
                energy_efficiency=energy_efficiency,
                throughput=throughput
            )
            
        except Exception as e:
            print(f"Error testing configuration: {e}")
            return None
    
    def _measure_baseline_performance(self, input_tensor: torch.Tensor, 
                                    expert_config: Dict[str, Any]) -> Dict[str, float]:
        """Measure baseline performance without optimization."""
        # Simulate baseline MoE computation
        batch_size, seq_len, d_model = input_tensor.shape
        
        # Estimate costs using kernel cost model
        attention_cost = self.kernel_cost_model.get_cost("attention_qk", batch_size)
        ffn_cost = self.kernel_cost_model.get_cost("ffn_gate", batch_size)
        moe_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        
        # Combine costs based on expert configuration
        top_k = expert_config.get("top_k", 2)
        expert_capacity = expert_config.get("expert_capacity", 0.7)
        
        total_energy = (
            attention_cost["energy_joules"] * 2 +  # QK and AV
            ffn_cost["energy_joules"] * top_k * expert_capacity +  # Expert computation
            moe_cost["energy_joules"]  # Routing overhead
        )
        
        total_latency = max(
            attention_cost["latency_ms"] * 2,
            ffn_cost["latency_ms"] * top_k * expert_capacity,
            moe_cost["latency_ms"]
        )
        
        # Estimate accuracy based on expert configuration
        accuracy = 0.85 + 0.1 * expert_capacity + 0.05 * (top_k - 1)
        accuracy = min(0.98, accuracy)  # Cap at 98%
        
        return {
            'energy_joules': total_energy,
            'latency_ms': total_latency,
            'power_watt': total_energy / (total_latency / 1000),
            'accuracy': accuracy,
            'thermal_imbalance': 0.1,  # Baseline thermal imbalance
            'memory_usage': 0.6
        }
    
    def _apply_optimization(self, target: OptimizationTarget, input_tensor: torch.Tensor,
                          expert_config: Dict[str, Any], baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply optimization based on target type."""
        optimized_metrics = baseline_metrics.copy()
        
        if target.target_type == "power":
            optimized_metrics = self._optimize_for_power(target, expert_config, optimized_metrics)
        elif target.target_type == "runtime":
            optimized_metrics = self._optimize_for_runtime(target, expert_config, optimized_metrics)
        elif target.target_type == "balanced":
            optimized_metrics = self._optimize_balanced(target, expert_config, optimized_metrics)
        elif target.target_type == "thermal":
            optimized_metrics = self._optimize_for_thermal(target, expert_config, optimized_metrics)
        
        return optimized_metrics
    
    def _optimize_for_power(self, target: OptimizationTarget, expert_config: Dict[str, Any],
                          metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize for power efficiency."""
        optimized = metrics.copy()
        
        # Power optimization strategies
        if metrics['power_watt'] > target.power_budget:
            # Reduce expert capacity to save power
            capacity_reduction = (metrics['power_watt'] - target.power_budget) / metrics['power_watt']
            new_capacity = max(0.3, expert_config.get("expert_capacity", 0.7) * (1 - capacity_reduction))
            
            # Adjust metrics
            optimized['energy_joules'] *= (new_capacity / expert_config.get("expert_capacity", 0.7))
            optimized['power_watt'] = optimized['energy_joules'] / (metrics['latency_ms'] / 1000)
            optimized['accuracy'] *= (0.8 + 0.2 * new_capacity)  # Accuracy impact
            
            # Slight latency increase due to reduced parallelism
            optimized['latency_ms'] *= 1.1
        
        return optimized
    
    def _optimize_for_runtime(self, target: OptimizationTarget, expert_config: Dict[str, Any],
                            metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize for runtime performance."""
        optimized = metrics.copy()
        
        # Runtime optimization strategies
        if metrics['latency_ms'] > target.latency_budget:
            # Increase expert capacity for better parallelism
            latency_reduction = (metrics['latency_ms'] - target.latency_budget) / metrics['latency_ms']
            new_capacity = min(1.0, expert_config.get("expert_capacity", 0.7) * (1 + latency_reduction))
            
            # Adjust metrics
            optimized['latency_ms'] *= (expert_config.get("expert_capacity", 0.7) / new_capacity)
            optimized['energy_joules'] *= (new_capacity / expert_config.get("expert_capacity", 0.7))
            optimized['power_watt'] = optimized['energy_joules'] / (optimized['latency_ms'] / 1000)
            optimized['accuracy'] *= (0.8 + 0.2 * new_capacity)  # Accuracy improvement
        
        return optimized
    
    def _optimize_balanced(self, target: OptimizationTarget, expert_config: Dict[str, Any],
                          metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize for balanced power and runtime."""
        optimized = metrics.copy()
        
        # Check both constraints
        power_violation = max(0, metrics['power_watt'] - target.power_budget) / target.power_budget
        latency_violation = max(0, metrics['latency_ms'] - target.latency_budget) / target.latency_budget
        
        if power_violation > 0 or latency_violation > 0:
            # Find optimal balance
            total_violation = power_violation + latency_violation
            
            # Adjust expert capacity based on which constraint is more violated
            if power_violation > latency_violation:
                # Power is more constrained
                capacity_adjustment = -power_violation * 0.5
            else:
                # Latency is more constrained
                capacity_adjustment = latency_violation * 0.3
            
            new_capacity = max(0.3, min(1.0, expert_config.get("expert_capacity", 0.7) + capacity_adjustment))
            
            # Adjust metrics proportionally
            capacity_ratio = new_capacity / expert_config.get("expert_capacity", 0.7)
            optimized['energy_joules'] *= capacity_ratio
            optimized['latency_ms'] *= (1 / capacity_ratio)
            optimized['power_watt'] = optimized['energy_joules'] / (optimized['latency_ms'] / 1000)
            optimized['accuracy'] *= (0.8 + 0.2 * new_capacity)
        
        return optimized
    
    def _optimize_for_thermal(self, target: OptimizationTarget, expert_config: Dict[str, Any],
                            metrics: Dict[str, float]) -> Dict[str, float]:
        """Optimize for thermal efficiency."""
        optimized = metrics.copy()
        
        # Thermal optimization strategies
        if metrics['thermal_imbalance'] > 0.2:  # High thermal imbalance
            # Reduce power to lower thermal stress
            thermal_reduction = metrics['thermal_imbalance'] * 0.3
            new_capacity = max(0.4, expert_config.get("expert_capacity", 0.7) * (1 - thermal_reduction))
            
            # Adjust metrics
            optimized['energy_joules'] *= (new_capacity / expert_config.get("expert_capacity", 0.7))
            optimized['power_watt'] = optimized['energy_joules'] / (metrics['latency_ms'] / 1000)
            optimized['thermal_imbalance'] *= 0.7  # Reduce thermal imbalance
            optimized['latency_ms'] *= 1.15  # Slight latency increase
            optimized['accuracy'] *= (0.8 + 0.2 * new_capacity)
        
        return optimized
    
    def _log_detailed_result(self, result: OptimizationResult):
        """Log detailed result for analysis."""
        log_entry = {
            'timestamp': time.time(),
            'target_type': result.target_type,
            'batch_size': result.batch_size,
            'expert_config': result.expert_config,
            'metrics': {
                'power_watt': result.achieved_power,
                'latency_ms': result.achieved_latency,
                'accuracy': result.achieved_accuracy,
                'thermal_imbalance': result.thermal_imbalance,
                'energy_efficiency': result.energy_efficiency,
                'throughput': result.throughput
            }
        }
        self.detailed_logs.append(log_entry)
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            'summary': self._generate_summary(),
            'power_vs_runtime_tradeoff': self._analyze_power_runtime_tradeoff(),
            'batch_size_analysis': self._analyze_batch_size_impact(),
            'expert_config_analysis': self._analyze_expert_config_impact(),
            'thermal_analysis': self._analyze_thermal_impact(),
            'efficiency_analysis': self._analyze_efficiency_metrics()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {}
        
        for target_type in set(r.target_type for r in self.results):
            target_results = [r for r in self.results if r.target_type == target_type]
            
            summary[target_type] = {
                'avg_power': np.mean([r.achieved_power for r in target_results]),
                'avg_latency': np.mean([r.achieved_latency for r in target_results]),
                'avg_accuracy': np.mean([r.achieved_accuracy for r in target_results]),
                'avg_energy_efficiency': np.mean([r.energy_efficiency for r in target_results]),
                'avg_throughput': np.mean([r.throughput for r in target_results]),
                'config_count': len(target_results)
            }
        
        return summary
    
    def _analyze_power_runtime_tradeoff(self) -> Dict[str, Any]:
        """Analyze the trade-off between power and runtime optimization."""
        power_results = [r for r in self.results if r.target_type == "power"]
        runtime_results = [r for r in self.results if r.target_type == "runtime"]
        balanced_results = [r for r in self.results if r.target_type == "balanced"]
        
        analysis = {
            'power_optimization': {
                'avg_power': np.mean([r.achieved_power for r in power_results]),
                'avg_latency': np.mean([r.achieved_latency for r in power_results]),
                'avg_efficiency': np.mean([r.energy_efficiency for r in power_results])
            },
            'runtime_optimization': {
                'avg_power': np.mean([r.achieved_power for r in runtime_results]),
                'avg_latency': np.mean([r.achieved_latency for r in runtime_results]),
                'avg_efficiency': np.mean([r.energy_efficiency for r in runtime_results])
            },
            'balanced_optimization': {
                'avg_power': np.mean([r.achieved_power for r in balanced_results]),
                'avg_latency': np.mean([r.achieved_latency for r in balanced_results]),
                'avg_efficiency': np.mean([r.energy_efficiency for r in balanced_results])
            }
        }
        
        # Calculate trade-off ratios
        power_runtime_ratio = analysis['power_optimization']['avg_power'] / analysis['runtime_optimization']['avg_power']
        latency_ratio = analysis['power_optimization']['avg_latency'] / analysis['runtime_optimization']['avg_latency']
        
        analysis['tradeoff_analysis'] = {
            'power_runtime_ratio': power_runtime_ratio,
            'latency_ratio': latency_ratio,
            'same_goal': abs(power_runtime_ratio - 1.0) < 0.1,  # Within 10%
            'power_favors_efficiency': analysis['power_optimization']['avg_efficiency'] > analysis['runtime_optimization']['avg_efficiency']
        }
        
        return analysis
    
    def _analyze_batch_size_impact(self) -> Dict[str, Any]:
        """Analyze impact of batch size on different optimization targets."""
        batch_analysis = {}
        
        for batch_size in self.batch_sizes:
            batch_results = [r for r in self.results if r.batch_size == batch_size]
            if not batch_results:
                continue
            
            batch_analysis[batch_size] = {
                'power_avg': np.mean([r.achieved_power for r in batch_results if r.target_type == "power"]),
                'runtime_avg': np.mean([r.achieved_latency for r in batch_results if r.target_type == "runtime"]),
                'efficiency_avg': np.mean([r.energy_efficiency for r in batch_results]),
                'throughput_avg': np.mean([r.throughput for r in batch_results])
            }
        
        return batch_analysis
    
    def _analyze_expert_config_impact(self) -> Dict[str, Any]:
        """Analyze impact of expert configuration on optimization."""
        config_analysis = {}
        
        for config in self.expert_configs:
            config_key = f"top_k_{config['top_k']}_cap_{config['expert_capacity']}"
            config_results = [r for r in self.results if r.expert_config == config]
            
            if config_results:
                config_analysis[config_key] = {
                    'avg_power': np.mean([r.achieved_power for r in config_results]),
                    'avg_latency': np.mean([r.achieved_latency for r in config_results]),
                    'avg_accuracy': np.mean([r.achieved_accuracy for r in config_results]),
                    'avg_efficiency': np.mean([r.energy_efficiency for r in config_results])
                }
        
        return config_analysis
    
    def _analyze_thermal_impact(self) -> Dict[str, Any]:
        """Analyze thermal impact on optimization."""
        thermal_results = [r for r in self.results if r.target_type == "thermal"]
        other_results = [r for r in self.results if r.target_type != "thermal"]
        
        return {
            'thermal_optimization': {
                'avg_thermal_imbalance': np.mean([r.thermal_imbalance for r in thermal_results]),
                'avg_power': np.mean([r.achieved_power for r in thermal_results]),
                'avg_efficiency': np.mean([r.energy_efficiency for r in thermal_results])
            },
            'non_thermal_optimization': {
                'avg_thermal_imbalance': np.mean([r.thermal_imbalance for r in other_results]),
                'avg_power': np.mean([r.achieved_power for r in other_results]),
                'avg_efficiency': np.mean([r.energy_efficiency for r in other_results])
            }
        }
    
    def _analyze_efficiency_metrics(self) -> Dict[str, Any]:
        """Analyze energy efficiency and throughput metrics."""
        efficiency_analysis = {}
        
        for target_type in set(r.target_type for r in self.results):
            target_results = [r for r in self.results if r.target_type == target_type]
            
            # Calculate efficiency distributions
            efficiencies = [r.energy_efficiency for r in target_results]
            throughputs = [r.throughput for r in target_results]
            
            efficiency_analysis[target_type] = {
                'efficiency_stats': {
                    'mean': np.mean(efficiencies),
                    'std': np.std(efficiencies),
                    'min': np.min(efficiencies),
                    'max': np.max(efficiencies),
                    'median': np.median(efficiencies)
                },
                'throughput_stats': {
                    'mean': np.mean(throughputs),
                    'std': np.std(throughputs),
                    'min': np.min(throughputs),
                    'max': np.max(throughputs),
                    'median': np.median(throughputs)
                }
            }
        
        return efficiency_analysis
    
    def save_results(self, filename: str = "dynamic_adaptation_results.json"):
        """Save results to file."""
        output_data = {
            'experiment_config': {
                'num_gpus': self.num_gpus,
                'num_experts': self.num_experts,
                'd_model': self.d_model,
                'optimization_targets': [t.__dict__ for t in self.optimization_targets],
                'batch_sizes': self.batch_sizes,
                'expert_configs': self.expert_configs
            },
            'results': [r.__dict__ for r in self.results],
            'analysis': self.analyze_results(),
            'detailed_logs': self.detailed_logs
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def plot_results(self, save_plots: bool = True):
        """Generate visualization plots."""
        if not self.results:
            print("No results to plot")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dynamic Adaptation Experiment Results', fontsize=16)
        
        # 1. Power vs Runtime trade-off
        ax1 = axes[0, 0]
        for target_type in ['power', 'runtime', 'balanced']:
            target_results = [r for r in self.results if r.target_type == target_type]
            if target_results:
                powers = [r.achieved_power for r in target_results]
                latencies = [r.achieved_latency for r in target_results]
                ax1.scatter(powers, latencies, label=target_type, alpha=0.7)
        
        ax1.set_xlabel('Power (W)')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Power vs Runtime Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy Efficiency by Target
        ax2 = axes[0, 1]
        target_types = list(set(r.target_type for r in self.results))
        efficiencies = []
        labels = []
        
        for target_type in target_types:
            target_results = [r for r in self.results if r.target_type == target_type]
            if target_results:
                efficiencies.append([r.energy_efficiency for r in target_results])
                labels.append(target_type)
        
        if efficiencies:
            ax2.boxplot(efficiencies, labels=labels)
            ax2.set_ylabel('Energy Efficiency (tokens/J)')
            ax2.set_title('Energy Efficiency by Optimization Target')
            ax2.grid(True, alpha=0.3)
        
        # 3. Batch Size Impact
        ax3 = axes[0, 2]
        batch_sizes = sorted(set(r.batch_size for r in self.results))
        avg_powers = []
        avg_latencies = []
        
        for batch_size in batch_sizes:
            batch_results = [r for r in self.results if r.batch_size == batch_size]
            if batch_results:
                avg_powers.append(np.mean([r.achieved_power for r in batch_results]))
                avg_latencies.append(np.mean([r.achieved_latency for r in batch_results]))
        
        if avg_powers:
            ax3_twin = ax3.twinx()
            line1 = ax3.plot(batch_sizes[:len(avg_powers)], avg_powers, 'b-', label='Power', marker='o')
            line2 = ax3_twin.plot(batch_sizes[:len(avg_latencies)], avg_latencies, 'r-', label='Latency', marker='s')
            
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Power (W)', color='b')
            ax3_twin.set_ylabel('Latency (ms)', color='r')
            ax3.set_title('Batch Size Impact')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left')
        
        # 4. Thermal Imbalance Analysis
        ax4 = axes[1, 0]
        thermal_results = [r for r in self.results if r.target_type == "thermal"]
        other_results = [r for r in self.results if r.target_type != "thermal"]
        
        if thermal_results and other_results:
            thermal_imbalances = [r.thermal_imbalance for r in thermal_results]
            other_imbalances = [r.thermal_imbalance for r in other_results]
            
            ax4.boxplot([thermal_imbalances, other_imbalances], 
                       labels=['Thermal Opt', 'Other Opt'])
            ax4.set_ylabel('Thermal Imbalance')
            ax4.set_title('Thermal Imbalance Comparison')
            ax4.grid(True, alpha=0.3)
        
        # 5. Throughput vs Energy Efficiency
        ax5 = axes[1, 1]
        for target_type in ['power', 'runtime', 'balanced']:
            target_results = [r for r in self.results if r.target_type == target_type]
            if target_results:
                throughputs = [r.throughput for r in target_results]
                efficiencies = [r.energy_efficiency for r in target_results]
                ax5.scatter(throughputs, efficiencies, label=target_type, alpha=0.7)
        
        ax5.set_xlabel('Throughput (tokens/s)')
        ax5.set_ylabel('Energy Efficiency (tokens/J)')
        ax5.set_title('Throughput vs Energy Efficiency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Accuracy vs Power Trade-off
        ax6 = axes[1, 2]
        for target_type in ['power', 'runtime', 'balanced']:
            target_results = [r for r in self.results if r.target_type == target_type]
            if target_results:
                accuracies = [r.achieved_accuracy for r in target_results]
                powers = [r.achieved_power for r in target_results]
                ax6.scatter(accuracies, powers, label=target_type, alpha=0.7)
        
        ax6.set_xlabel('Accuracy')
        ax6.set_ylabel('Power (W)')
        ax6.set_title('Accuracy vs Power Trade-off')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('dynamic_adaptation_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved to dynamic_adaptation_results.png")
        
        plt.show()

def main():
    """Run the dynamic adaptation experiment."""
    print("=== Dynamic Adaptation Experiment ===")
    print("Testing Power vs Runtime Optimization Trade-offs")
    
    # Create and run experiment
    experiment = DynamicAdaptationExperiment(num_gpus=8, num_experts=16, d_model=768)
    results = experiment.run_optimization_experiment()
    
    # Analyze results
    analysis = experiment.analyze_results()
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    
    # Power vs Runtime trade-off
    tradeoff_analysis = analysis['power_vs_runtime_tradeoff']['tradeoff_analysis']
    print(f"Power/Runtime Ratio: {tradeoff_analysis['power_runtime_ratio']:.3f}")
    print(f"Latency Ratio: {tradeoff_analysis['latency_ratio']:.3f}")
    print(f"Same Goal: {tradeoff_analysis['same_goal']}")
    print(f"Power Favors Efficiency: {tradeoff_analysis['power_favors_efficiency']}")
    
    # Summary by optimization target
    print("\n=== SUMMARY BY OPTIMIZATION TARGET ===")
    for target_type, summary in analysis['summary'].items():
        print(f"\n{target_type.upper()}:")
        print(f"  Avg Power: {summary['avg_power']:.1f}W")
        print(f"  Avg Latency: {summary['avg_latency']:.1f}ms")
        print(f"  Avg Energy Efficiency: {summary['avg_energy_efficiency']:.1f} tokens/J")
        print(f"  Avg Throughput: {summary['avg_throughput']:.1f} tokens/s")
    
    # Save results
    experiment.save_results()
    experiment.plot_results()
    
    print("\n=== EXPERIMENT COMPLETE ===")
    print("Results saved to dynamic_adaptation_results.json")
    print("Plots saved to dynamic_adaptation_results.png")

if __name__ == "__main__":
    main()