import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

@dataclass
class SyntheticWorkloadConfig:
    """Configuration for synthetic workload generation."""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    d_model: int
    num_experts: int
    noise_levels: List[float]
    error_margins: List[float]
    thermal_scenarios: List[str]
    memory_pressure_levels: List[float]

@dataclass
class SyntheticDataPoint:
    """Single synthetic data point for testing."""
    batch_size: int
    sequence_length: int
    d_model: int
    num_experts: int
    noise_level: float
    error_margin: float
    thermal_scenario: str
    memory_pressure: float
    input_tensor: torch.Tensor
    expected_output: torch.Tensor
    metadata: Dict[str, Any]

class SyntheticDataGenerator:
    """
    Generate synthetic data for testing MoE + TTT system with controlled noise and error margins.
    Addresses the need for synthetic testing mentioned in the notes.
    """
    def __init__(self, config: SyntheticWorkloadConfig):
        self.config = config
        self.generated_data = []
        self.noise_patterns = self._initialize_noise_patterns()
        
    def _initialize_noise_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize different noise patterns for testing."""
        patterns = {}
        
        # Gaussian noise patterns
        patterns['gaussian'] = np.random.normal(0, 1, 1000)
        
        # Systematic bias patterns
        patterns['systematic_bias'] = np.linspace(-0.5, 0.5, 1000)
        
        # Periodic noise patterns
        patterns['periodic'] = 0.3 * np.sin(np.linspace(0, 4*np.pi, 1000))
        
        # Spike noise patterns
        patterns['spike'] = np.zeros(1000)
        spike_indices = np.random.choice(1000, 50, replace=False)
        patterns['spike'][spike_indices] = np.random.normal(0, 2, 50)
        
        return patterns
    
    def generate_workload_dataset(self, num_samples: int = 1000) -> List[SyntheticDataPoint]:
        """Generate comprehensive synthetic workload dataset."""
        print(f"Generating {num_samples} synthetic data points...")
        
        for i in range(num_samples):
            # Randomly sample configuration parameters
            batch_size = np.random.choice(self.config.batch_sizes)
            seq_len = np.random.choice(self.config.sequence_lengths)
            noise_level = np.random.choice(self.config.noise_levels)
            error_margin = np.random.choice(self.config.error_margins)
            thermal_scenario = np.random.choice(self.config.thermal_scenarios)
            memory_pressure = np.random.choice(self.config.memory_pressure_levels)
            
            # Generate synthetic data point
            data_point = self._generate_single_datapoint(
                batch_size=batch_size,
                sequence_length=seq_len,
                noise_level=noise_level,
                error_margin=error_margin,
                thermal_scenario=thermal_scenario,
                memory_pressure=memory_pressure
            )
            
            self.generated_data.append(data_point)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} data points")
        
        return self.generated_data
    
    def _generate_single_datapoint(self, batch_size: int, sequence_length: int,
                                 noise_level: float, error_margin: float,
                                 thermal_scenario: str, memory_pressure: float) -> SyntheticDataPoint:
        """Generate a single synthetic data point."""
        
        # Generate base input tensor
        input_tensor = torch.randn(batch_size, sequence_length, self.config.d_model)
        
        # Apply noise based on noise level
        noisy_input = self._apply_noise_to_tensor(input_tensor, noise_level)
        
        # Generate expected output (simulate MoE computation)
        expected_output = self._simulate_moe_computation(noisy_input)
        
        # Generate metadata with error margins
        metadata = self._generate_metadata(
            batch_size, sequence_length, noise_level, error_margin,
            thermal_scenario, memory_pressure
        )
        
        return SyntheticDataPoint(
            batch_size=batch_size,
            sequence_length=sequence_length,
            d_model=self.config.d_model,
            num_experts=self.config.num_experts,
            noise_level=noise_level,
            error_margin=error_margin,
            thermal_scenario=thermal_scenario,
            memory_pressure=memory_pressure,
            input_tensor=noisy_input,
            expected_output=expected_output,
            metadata=metadata
        )
    
    def _apply_noise_to_tensor(self, tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Apply controlled noise to tensor."""
        if noise_level == 0:
            return tensor
        
        # Choose noise pattern
        noise_pattern = np.random.choice(list(self.noise_patterns.keys()))
        pattern = self.noise_patterns[noise_pattern]
        
        # Scale pattern to tensor size
        pattern_size = tensor.numel()
        if len(pattern) < pattern_size:
            # Repeat pattern if needed
            repeats = (pattern_size // len(pattern)) + 1
            pattern = np.tile(pattern, repeats)
        
        # Take first pattern_size elements
        pattern = pattern[:pattern_size].reshape(tensor.shape)
        
        # Apply noise
        noise_tensor = torch.tensor(pattern, dtype=tensor.dtype) * noise_level
        noisy_tensor = tensor + noise_tensor
        
        return noisy_tensor
    
    def _simulate_moe_computation(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Simulate MoE computation to generate expected output."""
        batch_size, seq_len, d_model = input_tensor.shape
        
        # Simulate attention mechanism
        attention_output = self._simulate_attention(input_tensor)
        
        # Simulate MoE routing and expert computation
        moe_output = self._simulate_moe_layer(attention_output)
        
        # Add residual connection
        output = attention_output + moe_output
        
        return output
    
    def _simulate_attention(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Simulate attention mechanism."""
        batch_size, seq_len, d_model = input_tensor.shape
        
        # Simple self-attention simulation
        # Q, K, V projections
        q = torch.randn(batch_size, seq_len, d_model // 8)  # 8 heads
        k = torch.randn(batch_size, seq_len, d_model // 8)
        v = torch.randn(batch_size, seq_len, d_model // 8)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_model // 8)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, v)
        
        # Project back to full dimension
        output = torch.randn(batch_size, seq_len, d_model)
        
        return output
    
    def _simulate_moe_layer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Simulate MoE layer computation."""
        batch_size, seq_len, d_model = input_tensor.shape
        
        # Simulate expert routing
        routing_logits = torch.randn(batch_size, seq_len, self.config.num_experts)
        routing_probs = torch.softmax(routing_logits, dim=-1)
        
        # Top-k expert selection
        top_k = min(2, self.config.num_experts)
        top_k_probs, top_k_indices = torch.topk(routing_probs, top_k, dim=-1)
        
        # Simulate expert computation
        expert_outputs = []
        for i in range(top_k):
            expert_output = torch.randn(batch_size, seq_len, d_model)
            expert_outputs.append(expert_output * top_k_probs[:, :, i:i+1])
        
        # Combine expert outputs
        combined_output = sum(expert_outputs)
        
        return combined_output
    
    def _generate_metadata(self, batch_size: int, sequence_length: int,
                          noise_level: float, error_margin: float,
                          thermal_scenario: str, memory_pressure: float) -> Dict[str, Any]:
        """Generate metadata with error margins."""
        
        # Base metrics
        base_energy = batch_size * sequence_length * 0.001  # Joules
        base_latency = batch_size * sequence_length * 0.01  # ms
        base_accuracy = 0.85 + 0.1 * (1 - noise_level)  # Higher noise = lower accuracy
        
        # Apply error margins
        energy_min = base_energy * (1 - error_margin)
        energy_max = base_energy * (1 + error_margin)
        latency_min = base_latency * (1 - error_margin)
        latency_max = base_latency * (1 + error_margin)
        accuracy_min = max(0.0, base_accuracy - error_margin)
        accuracy_max = min(1.0, base_accuracy + error_margin)
        
        # Thermal scenario effects
        thermal_effects = self._get_thermal_effects(thermal_scenario)
        
        # Memory pressure effects
        memory_effects = self._get_memory_effects(memory_pressure)
        
        metadata = {
            'base_metrics': {
                'energy_joules': base_energy,
                'latency_ms': base_latency,
                'accuracy': base_accuracy
            },
            'error_margins': {
                'energy_min': energy_min,
                'energy_max': energy_max,
                'latency_min': latency_min,
                'latency_max': latency_max,
                'accuracy_min': accuracy_min,
                'accuracy_max': accuracy_max
            },
            'thermal_scenario': thermal_scenario,
            'thermal_effects': thermal_effects,
            'memory_pressure': memory_pressure,
            'memory_effects': memory_effects,
            'noise_level': noise_level,
            'error_margin': error_margin,
            'confidence_level': 0.95
        }
        
        return metadata
    
    def _get_thermal_effects(self, thermal_scenario: str) -> Dict[str, float]:
        """Get thermal effects for different scenarios."""
        effects = {
            'normal': {
                'temperature': 60.0,
                'thermal_throttle': 0.0,
                'power_factor': 1.0,
                'latency_factor': 1.0
            },
            'hot': {
                'temperature': 85.0,
                'thermal_throttle': 0.3,
                'power_factor': 1.2,
                'latency_factor': 1.5
            },
            'imbalanced': {
                'temperature': 75.0,
                'thermal_throttle': 0.1,
                'power_factor': 1.1,
                'latency_factor': 1.2
            },
            'cool': {
                'temperature': 40.0,
                'thermal_throttle': 0.0,
                'power_factor': 0.9,
                'latency_factor': 0.95
            }
        }
        
        return effects.get(thermal_scenario, effects['normal'])
    
    def _get_memory_effects(self, memory_pressure: float) -> Dict[str, float]:
        """Get memory pressure effects."""
        return {
            'memory_utilization': memory_pressure,
            'bandwidth_factor': max(0.5, 1.0 - memory_pressure * 0.5),
            'latency_factor': 1.0 + memory_pressure * 0.3,
            'energy_factor': 1.0 + memory_pressure * 0.2
        }
    
    def analyze_generated_data(self) -> Dict[str, Any]:
        """Analyze the generated synthetic data."""
        if not self.generated_data:
            return {"error": "No data to analyze"}
        
        analysis = {
            'data_summary': self._generate_data_summary(),
            'noise_analysis': self._analyze_noise_effects(),
            'error_margin_analysis': self._analyze_error_margins(),
            'thermal_analysis': self._analyze_thermal_effects(),
            'memory_analysis': self._analyze_memory_effects(),
            'correlation_analysis': self._analyze_correlations()
        }
        
        return analysis
    
    def _generate_data_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for generated data."""
        summary = {
            'total_samples': len(self.generated_data),
            'batch_size_distribution': defaultdict(int),
            'noise_level_distribution': defaultdict(int),
            'error_margin_distribution': defaultdict(int),
            'thermal_scenario_distribution': defaultdict(int),
            'memory_pressure_distribution': defaultdict(int)
        }
        
        for data_point in self.generated_data:
            summary['batch_size_distribution'][data_point.batch_size] += 1
            summary['noise_level_distribution'][data_point.noise_level] += 1
            summary['error_margin_distribution'][data_point.error_margin] += 1
            summary['thermal_scenario_distribution'][data_point.thermal_scenario] += 1
            summary['memory_pressure_distribution'][data_point.memory_pressure] += 1
        
        return summary
    
    def _analyze_noise_effects(self) -> Dict[str, Any]:
        """Analyze effects of different noise levels."""
        noise_analysis = {}
        
        for noise_level in self.config.noise_levels:
            noise_data = [dp for dp in self.generated_data if dp.noise_level == noise_level]
            
            if noise_data:
                accuracies = [dp.metadata['base_metrics']['accuracy'] for dp in noise_data]
                energies = [dp.metadata['base_metrics']['energy_joules'] for dp in noise_data]
                latencies = [dp.metadata['base_metrics']['latency_ms'] for dp in noise_data]
                
                noise_analysis[noise_level] = {
                    'avg_accuracy': np.mean(accuracies),
                    'avg_energy': np.mean(energies),
                    'avg_latency': np.mean(latencies),
                    'sample_count': len(noise_data)
                }
        
        return noise_analysis
    
    def _analyze_error_margins(self) -> Dict[str, Any]:
        """Analyze effects of different error margins."""
        margin_analysis = {}
        
        for error_margin in self.config.error_margins:
            margin_data = [dp for dp in self.generated_data if dp.error_margin == error_margin]
            
            if margin_data:
                energy_ranges = []
                latency_ranges = []
                accuracy_ranges = []
                
                for dp in margin_data:
                    energy_ranges.append(
                        dp.metadata['error_margins']['energy_max'] - dp.metadata['error_margins']['energy_min']
                    )
                    latency_ranges.append(
                        dp.metadata['error_margins']['latency_max'] - dp.metadata['error_margins']['latency_min']
                    )
                    accuracy_ranges.append(
                        dp.metadata['error_margins']['accuracy_max'] - dp.metadata['error_margins']['accuracy_min']
                    )
                
                margin_analysis[error_margin] = {
                    'avg_energy_range': np.mean(energy_ranges),
                    'avg_latency_range': np.mean(latency_ranges),
                    'avg_accuracy_range': np.mean(accuracy_ranges),
                    'sample_count': len(margin_data)
                }
        
        return margin_analysis
    
    def _analyze_thermal_effects(self) -> Dict[str, Any]:
        """Analyze effects of different thermal scenarios."""
        thermal_analysis = {}
        
        for scenario in self.config.thermal_scenarios:
            thermal_data = [dp for dp in self.generated_data if dp.thermal_scenario == scenario]
            
            if thermal_data:
                power_factors = [dp.metadata['thermal_effects']['power_factor'] for dp in thermal_data]
                latency_factors = [dp.metadata['thermal_effects']['latency_factor'] for dp in thermal_data]
                temperatures = [dp.metadata['thermal_effects']['temperature'] for dp in thermal_data]
                
                thermal_analysis[scenario] = {
                    'avg_power_factor': np.mean(power_factors),
                    'avg_latency_factor': np.mean(latency_factors),
                    'avg_temperature': np.mean(temperatures),
                    'sample_count': len(thermal_data)
                }
        
        return thermal_analysis
    
    def _analyze_memory_effects(self) -> Dict[str, Any]:
        """Analyze effects of different memory pressure levels."""
        memory_analysis = {}
        
        for pressure in self.config.memory_pressure_levels:
            memory_data = [dp for dp in self.generated_data if dp.memory_pressure == pressure]
            
            if memory_data:
                bandwidth_factors = [dp.metadata['memory_effects']['bandwidth_factor'] for dp in memory_data]
                latency_factors = [dp.metadata['memory_effects']['latency_factor'] for dp in memory_data]
                energy_factors = [dp.metadata['memory_effects']['energy_factor'] for dp in memory_data]
                
                memory_analysis[pressure] = {
                    'avg_bandwidth_factor': np.mean(bandwidth_factors),
                    'avg_latency_factor': np.mean(latency_factors),
                    'avg_energy_factor': np.mean(energy_factors),
                    'sample_count': len(memory_data)
                }
        
        return memory_analysis
    
    def _analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between different parameters."""
        correlations = {}
        
        # Extract numerical parameters
        noise_levels = [dp.noise_level for dp in self.generated_data]
        error_margins = [dp.error_margin for dp in self.generated_data]
        memory_pressures = [dp.memory_pressure for dp in self.generated_data]
        accuracies = [dp.metadata['base_metrics']['accuracy'] for dp in self.generated_data]
        energies = [dp.metadata['base_metrics']['energy_joules'] for dp in self.generated_data]
        latencies = [dp.metadata['base_metrics']['latency_ms'] for dp in self.generated_data]
        
        # Calculate correlations
        correlations['noise_accuracy'] = np.corrcoef(noise_levels, accuracies)[0, 1]
        correlations['noise_energy'] = np.corrcoef(noise_levels, energies)[0, 1]
        correlations['memory_latency'] = np.corrcoef(memory_pressures, latencies)[0, 1]
        correlations['memory_energy'] = np.corrcoef(memory_pressures, energies)[0, 1]
        correlations['energy_latency'] = np.corrcoef(energies, latencies)[0, 1]
        
        return correlations
    
    def save_dataset(self, filename: str = "synthetic_moe_dataset.json"):
        """Save generated dataset to file."""
        dataset = {
            'config': self.config.__dict__,
            'data_points': []
        }
        
        for dp in self.generated_data:
            data_point_dict = {
                'batch_size': dp.batch_size,
                'sequence_length': dp.sequence_length,
                'd_model': dp.d_model,
                'num_experts': dp.num_experts,
                'noise_level': dp.noise_level,
                'error_margin': dp.error_margin,
                'thermal_scenario': dp.thermal_scenario,
                'memory_pressure': dp.memory_pressure,
                'input_tensor_shape': list(dp.input_tensor.shape),
                'expected_output_shape': list(dp.expected_output.shape),
                'metadata': dp.metadata
            }
            dataset['data_points'].append(data_point_dict)
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Synthetic dataset saved to {filename}")
    
    def plot_analysis(self, save_plots: bool = True):
        """Generate visualization plots for analysis."""
        if not self.generated_data:
            print("No data to plot")
            return
        
        analysis = self.analyze_generated_data()
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synthetic Data Analysis', fontsize=16)
        
        # 1. Noise Level Effects
        ax1 = axes[0, 0]
        noise_analysis = analysis['noise_analysis']
        noise_levels = list(noise_analysis.keys())
        accuracies = [noise_analysis[nl]['avg_accuracy'] for nl in noise_levels]
        
        ax1.plot(noise_levels, accuracies, 'o-', label='Accuracy')
        ax1.set_xlabel('Noise Level')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Noise Level vs Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # 2. Error Margin Effects
        ax2 = axes[0, 1]
        margin_analysis = analysis['error_margin_analysis']
        margins = list(margin_analysis.keys())
        energy_ranges = [margin_analysis[m]['avg_energy_range'] for m in margins]
        
        ax2.plot(margins, energy_ranges, 's-', label='Energy Range')
        ax2.set_xlabel('Error Margin')
        ax2.set_ylabel('Average Energy Range')
        ax2.set_title('Error Margin vs Energy Uncertainty')
        ax2.grid(True, alpha=0.3)
        
        # 3. Thermal Scenario Effects
        ax3 = axes[0, 2]
        thermal_analysis = analysis['thermal_analysis']
        scenarios = list(thermal_analysis.keys())
        power_factors = [thermal_analysis[s]['avg_power_factor'] for s in scenarios]
        latency_factors = [thermal_analysis[s]['avg_latency_factor'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax3.bar(x - width/2, power_factors, width, label='Power Factor')
        ax3.bar(x + width/2, latency_factors, width, label='Latency Factor')
        ax3.set_xlabel('Thermal Scenario')
        ax3.set_ylabel('Factor')
        ax3.set_title('Thermal Effects')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Memory Pressure Effects
        ax4 = axes[1, 0]
        memory_analysis = analysis['memory_analysis']
        pressures = list(memory_analysis.keys())
        bandwidth_factors = [memory_analysis[p]['avg_bandwidth_factor'] for p in pressures]
        
        ax4.plot(pressures, bandwidth_factors, '^-', label='Bandwidth Factor')
        ax4.set_xlabel('Memory Pressure')
        ax4.set_ylabel('Bandwidth Factor')
        ax4.set_title('Memory Pressure vs Bandwidth')
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlation Matrix
        ax5 = axes[1, 1]
        correlations = analysis['correlation_analysis']
        corr_names = list(correlations.keys())
        corr_values = list(correlations.values())
        
        bars = ax5.bar(corr_names, corr_values)
        ax5.set_ylabel('Correlation Coefficient')
        ax5.set_title('Parameter Correlations')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Color bars by correlation strength
        for bar, value in zip(bars, corr_values):
            if abs(value) > 0.5:
                bar.set_color('red')
            elif abs(value) > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        # 6. Data Distribution
        ax6 = axes[1, 2]
        summary = analysis['data_summary']
        batch_sizes = list(summary['batch_size_distribution'].keys())
        batch_counts = list(summary['batch_size_distribution'].values())
        
        ax6.pie(batch_counts, labels=batch_sizes, autopct='%1.1f%%')
        ax6.set_title('Batch Size Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('synthetic_data_analysis.png', dpi=300, bbox_inches='tight')
            print("Analysis plots saved to synthetic_data_analysis.png")
        
        plt.show()

def main():
    """Generate and analyze synthetic data."""
    print("=== Synthetic Data Generation for MoE + TTT Testing ===")
    
    # Configuration
    config = SyntheticWorkloadConfig(
        batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        sequence_lengths=[64, 128, 256, 512, 1024],
        d_model=768,
        num_experts=16,
        noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2],
        error_margins=[0.05, 0.1, 0.15, 0.2],
        thermal_scenarios=['normal', 'hot', 'imbalanced', 'cool'],
        memory_pressure_levels=[0.3, 0.5, 0.7, 0.9]
    )
    
    # Generate data
    generator = SyntheticDataGenerator(config)
    dataset = generator.generate_workload_dataset(num_samples=500)
    
    # Analyze data
    analysis = generator.analyze_generated_data()
    
    # Print key findings
    print("\n=== SYNTHETIC DATA ANALYSIS ===")
    
    print(f"Total samples generated: {len(dataset)}")
    print(f"Configuration space: {len(config.batch_sizes)} batch sizes × {len(config.sequence_lengths)} seq lengths × {len(config.noise_levels)} noise levels")
    
    # Noise effects
    noise_analysis = analysis['noise_analysis']
    print(f"\nNoise Effects:")
    for noise_level, stats in noise_analysis.items():
        print(f"  Noise {noise_level}: Accuracy = {stats['avg_accuracy']:.3f}")
    
    # Error margin effects
    margin_analysis = analysis['error_margin_analysis']
    print(f"\nError Margin Effects:")
    for margin, stats in margin_analysis.items():
        print(f"  Margin {margin}: Energy Range = {stats['avg_energy_range']:.6f}")
    
    # Correlations
    correlations = analysis['correlation_analysis']
    print(f"\nKey Correlations:")
    for name, value in correlations.items():
        print(f"  {name}: {value:.3f}")
    
    # Save dataset and plots
    generator.save_dataset()
    generator.plot_analysis()
    
    print("\n=== SYNTHETIC DATA GENERATION COMPLETE ===")
    print("Dataset saved to synthetic_moe_dataset.json")
    print("Analysis plots saved to synthetic_data_analysis.png")

if __name__ == "__main__":
    main() 