#!/usr/bin/env python3
"""
Single GPU TTT Comparison Experiment

This script compares regular TTT vs EnergyAwareTTTRouter on a single GPU
with a small transformer, monitoring energy, power, and hardware metrics.

Focus: Practical single-GPU testing with hardware monitoring
"""

import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
from collections import deque

# Import our custom modules
from src.moe_models import MoEConfig, MoETransformerBlock
from src.routers import EnergyAwareTTTRouter, AdaptiveRouter, RoutingStrategy
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.metrics_logger import MetricsLogger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HardwareMetrics:
    """Hardware metrics for comparison."""
    timestamp: float
    temperature: float
    power_watt: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    inference_latency_ms: float
    energy_per_token_j: float
    throughput_tokens_per_sec: float


class SimpleTransformer(nn.Module):
    """
    Simple transformer for single GPU testing.
    Small enough to fit on one GPU but complex enough to show TTT benefits.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 vocab_size: int = 10000,
                 seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_length, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # TTT fast weights (for regular TTT)
        self.ttt_fast_weights = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.ttt_optimizer = torch.optim.AdamW([self.ttt_fast_weights], lr=1e-4)
        
        # TTT buffers
        self.ttt_buffer = []
        self.ttt_update_frequency = 128  # Update every 128 tokens
        
        logger.info(f"Created SimpleTransformer: {d_model}d, {num_layers} layers, {num_heads} heads")
    
    def _create_transformer_layer(self, d_model: int, num_heads: int, dropout: float) -> nn.Module:
        """Create a single transformer layer."""
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
    
    def forward(self, input_ids: torch.Tensor, use_ttt: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with optional TTT."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Apply TTT fast weights if enabled
        if use_ttt:
            hidden_states = hidden_states + torch.tanh(hidden_states @ self.ttt_fast_weights)
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            
            # Apply TTT updates periodically
            if use_ttt and i == len(self.layers) - 1:  # Last layer
                self._update_ttt_buffers(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def _update_ttt_buffers(self, hidden_states: torch.Tensor):
        """Update TTT buffers for fast weight adaptation."""
        self.ttt_buffer.append(hidden_states.detach())
        
        # Perform TTT update when buffer is full
        if len(self.ttt_buffer) >= self.ttt_update_frequency:
            self._perform_ttt_update()
    
    def _perform_ttt_update(self):
        """Perform TTT update on fast weights."""
        if not self.ttt_buffer:
            return
        
        # Simple TTT loss: minimize variance of hidden states
        buffer_tensor = torch.stack(self.ttt_buffer)
        target = buffer_tensor.mean(dim=0)  # Target: mean of recent hidden states
        
        # Compute loss
        loss = F.mse_loss(buffer_tensor, target.unsqueeze(0).expand_as(buffer_tensor))
        
        # Update fast weights
        self.ttt_optimizer.zero_grad()
        loss.backward()
        self.ttt_optimizer.step()
        
        # Clear buffer
        self.ttt_buffer.clear()


class MoETransformer(nn.Module):
    """
    MoE transformer with EnergyAwareTTTRouter for comparison.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_experts: int = 4,
                 top_k: int = 2,
                 num_heads: int = 8,
                 vocab_size: int = 10000,
                 seq_length: int = 512,
                 kernel_cost_model: KernelCostModel = None,
                 gpu_monitor: GpuSystemMonitor = None):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_length = seq_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_length, d_model)
        
        # MoE configuration
        moe_config = MoEConfig(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
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
        
        # Create MoE layers with EnergyAwareTTTRouter
        self.moe_layers = nn.ModuleList()
        for _ in range(num_layers):
            moe_layer = MoETransformerBlock(
                moe_config,
                kernel_cost_model or KernelCostModel(),
                gpu_monitor or GpuSystemMonitor(device_id=0)
            )
            
            # Replace router with EnergyAwareTTTRouter
            moe_layer.router = EnergyAwareTTTRouter(
                config=moe_config,
                kernel_cost_model=kernel_cost_model or KernelCostModel(),
                gpu_system_monitor=gpu_monitor or GpuSystemMonitor(device_id=0),
                ttt_chunk_size=512,  # Smaller chunks for single GPU
                ttt_update_frequency=128,
                energy_aware_lr=1e-4,
                muon_enabled=True
            )
            
            self.moe_layers.append(moe_layer)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        logger.info(f"Created MoETransformer: {d_model}d, {num_layers} layers, {num_experts} experts")
    
    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with MoE layers."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # MoE layers
        for moe_layer in self.moe_layers:
            hidden_states = moe_layer(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }


class SingleGPUTTTComparison:
    """
    Single GPU comparison between regular TTT and EnergyAwareTTTRouter.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(
            data_path=args.kernel_cost_model_json,
            gpu_type=args.gpu_type
        )
        self.gpu_monitor = GpuSystemMonitor(device_id=args.device_id)
        self.metrics_logger = MetricsLogger(args.log_file)
        
        # Create models
        self.simple_transformer = SimpleTransformer(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            dropout=args.dropout
        ).to(self.device)
        
        self.moe_transformer = MoETransformer(
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_experts=args.num_experts,
            top_k=args.top_k,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            kernel_cost_model=self.kernel_cost_model,
            gpu_monitor=self.gpu_monitor
        ).to(self.device)
        
        # Results storage
        self.results = {
            'regular_ttt': [],
            'energy_aware_ttt': []
        }
        
        logger.info(f"Initialized SingleGPUTTTComparison on device: {self.device}")
    
    def generate_synthetic_data(self, batch_size: int, seq_length: int) -> torch.Tensor:
        """Generate synthetic input data for testing."""
        return torch.randint(0, self.args.vocab_size, (batch_size, seq_length)).to(self.device)
    
    def measure_hardware_metrics(self, model: nn.Module, input_ids: torch.Tensor, 
                               model_name: str, use_ttt: bool = False) -> HardwareMetrics:
        """Measure hardware metrics for a single forward pass."""
        batch_size, seq_length = input_ids.shape
        num_tokens = batch_size * seq_length
        
        # Get initial hardware state
        initial_stats = self.gpu_monitor.get_current_stats()
        initial_temp = initial_stats.get('temperature', 50.0)
        initial_power = initial_stats.get('power_watt', 200.0)
        initial_memory = initial_stats.get('memory_utilization_percent', 50.0)
        
        # Warm-up run
        with torch.no_grad():
            if model_name == 'simple':
                _ = model(input_ids, use_ttt=use_ttt)
            else:
                _ = model(input_ids)
        
        # Measure forward pass
        start_time = time.time()
        with torch.no_grad():
            if model_name == 'simple':
                outputs = model(input_ids, use_ttt=use_ttt)
            else:
                outputs = model(input_ids)
        end_time = time.time()
        
        # Get final hardware state
        final_stats = self.gpu_monitor.get_current_stats()
        final_temp = final_stats.get('temperature', 50.0)
        final_power = final_stats.get('power_watt', 200.0)
        final_memory = final_stats.get('memory_utilization_percent', 50.0)
        
        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        avg_power = (initial_power + final_power) / 2
        energy_per_token = (avg_power * latency_ms / 1000) / num_tokens
        throughput = num_tokens / (latency_ms / 1000)
        
        return HardwareMetrics(
            timestamp=time.time(),
            temperature=final_temp,
            power_watt=avg_power,
            memory_utilization_percent=(initial_memory + final_memory) / 2,
            gpu_utilization_percent=final_stats.get('gpu_utilization_percent', 0.0),
            inference_latency_ms=latency_ms,
            energy_per_token_j=energy_per_token,
            throughput_tokens_per_sec=throughput
        )
    
    def run_comparison(self):
        """Run the comparison experiment."""
        logger.info("Starting Single GPU TTT Comparison")
        
        # Generate test data
        test_data = self.generate_synthetic_data(
            batch_size=self.args.batch_size,
            seq_length=self.args.seq_length
        )
        
        # Test scenarios
        scenarios = [
            ('regular_ttt', self.simple_transformer, True),   # Regular TTT
            ('energy_aware_ttt', self.moe_transformer, False)  # EnergyAwareTTTRouter
        ]
        
        for scenario_name, model, use_ttt in scenarios:
            logger.info(f"Testing {scenario_name}")
            
            scenario_results = []
            
            for run in range(self.args.num_runs):
                # Measure hardware metrics
                metrics = self.measure_hardware_metrics(
                    model, test_data, 
                    'simple' if model == self.simple_transformer else 'moe',
                    use_ttt
                )
                
                scenario_results.append(metrics)
                
                # Print progress
                if run % self.args.log_interval == 0:
                    logger.info(f"  Run {run}: Latency={metrics.inference_latency_ms:.2f}ms, "
                               f"Power={metrics.power_watt:.1f}W, Temp={metrics.temperature:.1f}°C")
                
                # Small delay between runs
                time.sleep(0.1)
            
            self.results[scenario_name] = scenario_results
        
        # Analyze results
        self._analyze_results()
        self._generate_plots()
        self._save_results()
    
    def _analyze_results(self):
        """Analyze and compare results."""
        logger.info("Analyzing comparison results")
        
        # Calculate summary statistics
        summary_data = []
        
        for scenario_name, results in self.results.items():
            if not results:
                continue
            
            # Calculate averages
            avg_latency = np.mean([r.inference_latency_ms for r in results])
            avg_power = np.mean([r.power_watt for r in results])
            avg_energy = np.mean([r.energy_per_token_j for r in results])
            avg_temp = np.mean([r.temperature for r in results])
            avg_throughput = np.mean([r.throughput_tokens_per_sec for r in results])
            avg_memory = np.mean([r.memory_utilization_percent for r in results])
            
            # Calculate standard deviations
            std_latency = np.std([r.inference_latency_ms for r in results])
            std_power = np.std([r.power_watt for r in results])
            std_energy = np.std([r.energy_per_token_j for r in results])
            
            summary_data.append({
                'Scenario': scenario_name,
                'Avg Latency (ms)': avg_latency,
                'Std Latency (ms)': std_latency,
                'Avg Power (W)': avg_power,
                'Std Power (W)': std_power,
                'Avg Energy (J/token)': avg_energy,
                'Std Energy (J/token)': std_energy,
                'Avg Temperature (°C)': avg_temp,
                'Avg Throughput (tokens/sec)': avg_throughput,
                'Avg Memory (%)': avg_memory
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Calculate improvements
        if len(summary_data) >= 2:
            regular_ttt = summary_data[0]
            energy_aware = summary_data[1]
            
            improvements = {
                'latency_improvement_pct': ((regular_ttt['Avg Latency (ms)'] - energy_aware['Avg Latency (ms)']) / 
                                          regular_ttt['Avg Latency (ms)']) * 100,
                'power_improvement_pct': ((regular_ttt['Avg Power (W)'] - energy_aware['Avg Power (W)']) / 
                                        regular_ttt['Avg Power (W)']) * 100,
                'energy_improvement_pct': ((regular_ttt['Avg Energy (J/token)'] - energy_aware['Avg Energy (J/token)']) / 
                                         regular_ttt['Avg Energy (J/token)']) * 100,
                'throughput_improvement_pct': ((energy_aware['Avg Throughput (tokens/sec)'] - regular_ttt['Avg Throughput (tokens/sec)']) / 
                                             regular_ttt['Avg Throughput (tokens/sec)']) * 100
            }
            
            self.improvements = improvements
            
            # Print results
            logger.info("\n=== Comparison Results ===")
            logger.info(self.summary_df.to_string(index=False))
            
            logger.info("\n=== Improvements ===")
            logger.info(f"Latency improvement: {improvements['latency_improvement_pct']:.2f}%")
            logger.info(f"Power improvement: {improvements['power_improvement_pct']:.2f}%")
            logger.info(f"Energy improvement: {improvements['energy_improvement_pct']:.2f}%")
            logger.info(f"Throughput improvement: {improvements['throughput_improvement_pct']:.2f}%")
    
    def _generate_plots(self):
        """Generate comparison plots."""
        logger.info("Generating comparison plots")
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Single GPU TTT Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        scenarios = list(self.results.keys())
        metrics = ['inference_latency_ms', 'power_watt', 'energy_per_token_j', 
                  'temperature', 'throughput_tokens_per_sec', 'memory_utilization_percent']
        metric_names = ['Latency (ms)', 'Power (W)', 'Energy (J/token)', 
                       'Temperature (°C)', 'Throughput (tokens/sec)', 'Memory (%)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 3, i % 3
            
            # Extract data for this metric
            data = []
            labels = []
            for scenario in scenarios:
                if scenario in self.results:
                    values = [getattr(r, metric) for r in self.results[scenario]]
                    data.append(values)
                    labels.append(scenario.replace('_', ' ').title())
            
            # Create box plot
            axes[row, col].boxplot(data, labels=labels)
            axes[row, col].set_title(metric_name)
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'single_gpu_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create time series plot
        self._create_time_series_plot(output_dir)
        
        logger.info(f"Plots saved to {output_dir}")
    
    def _create_time_series_plot(self, output_dir: Path):
        """Create time series plot of metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Hardware Metrics Over Time', fontsize=14)
        
        metrics = ['power_watt', 'temperature', 'inference_latency_ms', 'energy_per_token_j']
        metric_names = ['Power (W)', 'Temperature (°C)', 'Latency (ms)', 'Energy (J/token)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 2, i % 2
            
            for scenario, results in self.results.items():
                if results:
                    timestamps = [r.timestamp - results[0].timestamp for r in results]
                    values = [getattr(r, metric) for r in results]
                    
                    axes[row, col].plot(timestamps, values, 
                                      label=scenario.replace('_', ' ').title(),
                                      marker='o', markersize=3)
            
            axes[row, col].set_title(metric_name)
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save detailed results to files."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary table
        if hasattr(self, 'summary_df'):
            self.summary_df.to_csv(output_dir / 'comparison_summary.csv', index=False)
        
        # Save detailed results
        detailed_results = {}
        for scenario_name, results in self.results.items():
            detailed_results[scenario_name] = [
                {
                    'timestamp': r.timestamp,
                    'temperature': r.temperature,
                    'power_watt': r.power_watt,
                    'memory_utilization_percent': r.memory_utilization_percent,
                    'gpu_utilization_percent': r.gpu_utilization_percent,
                    'inference_latency_ms': r.inference_latency_ms,
                    'energy_per_token_j': r.energy_per_token_j,
                    'throughput_tokens_per_sec': r.throughput_tokens_per_sec
                }
                for r in results
            ]
        
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save improvements
        if hasattr(self, 'improvements'):
            with open(output_dir / 'improvements.json', 'w') as f:
                json.dump(self.improvements, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Single GPU TTT Comparison")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # MoE configuration
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k expert selection")
    
    # Experiment configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs per scenario")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    # System configuration
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--gpu_type", type=str, default="A100", help="GPU type")
    parser.add_argument("--kernel_cost_model_json", type=str, default="energy/cost_table.json")
    parser.add_argument("--log_file", type=str, default="logs/single_gpu_ttt_comparison.log")
    parser.add_argument("--output_dir", type=str, default="results/single_gpu_comparison")
    
    args = parser.parse_args()
    
    # Run comparison
    comparison = SingleGPUTTTComparison(args)
    comparison.run_comparison()


if __name__ == "__main__":
    main() 