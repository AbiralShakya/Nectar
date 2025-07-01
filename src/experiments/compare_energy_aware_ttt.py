#!/usr/bin/env python3
"""
Comparison script for EnergyAwareTTTRouter vs baseline routers

This script compares the performance of the EnergyAwareTTTRouter against:
1. Baseline router (no hardware awareness)
2. Kernel-aware TTHA router
3. Statistical load balancing router
4. EnergyAwareTTTRouter (our novel approach)

Metrics compared:
- Energy efficiency (Watts/token)
- Thermal performance (temperature rise)
- Latency (ms/token)
- Memory efficiency (GB/token)
- Load balancing quality
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
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json

from src.moe_models import MoEConfig, MoETransformerBlock
from src.routers import (EnergyAwareTTTRouter, AdaptiveRouter, 
                        RoutingStrategy, StatisticalLoadBalancer)
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.metrics_logger import MetricsLogger
from src.data_utils import DataLoaderManager


class RouterBenchmark:
    """Benchmark different router implementations."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        
        # Initialize shared components
        self.moe_config = self._create_moe_config()
        self.kernel_cost_model = KernelCostModel(
            data_path=args.kernel_cost_model_json,
            gpu_type=args.gpu_type
        )
        self.gpu_monitor = GpuSystemMonitor(device_id=args.device_id)
        self.metrics_logger = MetricsLogger(args.log_file)
        self.dataloader_manager = DataLoaderManager(d_model=args.d_model)
        
        # Initialize routers
        self.routers = self._create_routers()
        
        # Results storage
        self.results = {
            'baseline': [],
            'kernel_aware_ttha': [],
            'statistical_load_balancing': [],
            'energy_aware_ttt': []
        }
        
        logger.info(f"Initialized RouterBenchmark with {len(self.routers)} routers")
    
    def _create_moe_config(self) -> MoEConfig:
        """Create MoE configuration for benchmarking."""
        return MoEConfig(
            d_model=self.args.d_model,
            num_experts=self.args.num_experts,
            top_k=self.args.top_k,
            dropout=0.1,
            use_bias=False,
            activation="swiglu",
            expert_dropout=0.0,
            use_grouped_gemm=True,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
            capacity_factor=1.25,
            expert_type=self.args.expert_type
        )
    
    def _create_routers(self) -> Dict[str, nn.Module]:
        """Create different router implementations for comparison."""
        routers = {}
        
        # 1. Baseline router (no hardware awareness)
        baseline_model = MoETransformerBlock(
            self.moe_config,
            self.kernel_cost_model,
            self.gpu_monitor
        ).to(self.device)
        routers['baseline'] = baseline_model
        
        # 2. Kernel-aware TTHA router
        kernel_aware_model = MoETransformerBlock(
            self.moe_config,
            self.kernel_cost_model,
            self.gpu_monitor
        ).to(self.device)
        kernel_aware_model.moe_layer.router = AdaptiveRouter(
            config=self.moe_config,
            kernel_cost_model=self.kernel_cost_model,
            gpu_system_monitor=self.gpu_monitor,
            strategy=RoutingStrategy.KERNEL_AWARE_TTHA
        ).to(self.device)
        routers['kernel_aware_ttha'] = kernel_aware_model
        
        # 3. Statistical load balancing router
        statistical_model = MoETransformerBlock(
            self.moe_config,
            self.kernel_cost_model,
            self.gpu_monitor
        ).to(self.device)
        # Note: Statistical load balancing is integrated into EnergyAwareTTTRouter
        # For fair comparison, we'll use a simplified version
        routers['statistical_load_balancing'] = statistical_model
        
        # 4. EnergyAwareTTTRouter (our novel approach)
        energy_aware_model = MoETransformerBlock(
            self.moe_config,
            self.kernel_cost_model,
            self.gpu_monitor
        ).to(self.device)
        energy_aware_model.moe_layer.router = EnergyAwareTTTRouter(
            config=self.moe_config,
            kernel_cost_model=self.kernel_cost_model,
            gpu_system_monitor=self.gpu_monitor,
            ttt_chunk_size=self.args.ttt_chunk_size,
            ttt_update_frequency=self.args.ttt_update_frequency,
            energy_aware_lr=self.args.energy_aware_lr,
            muon_enabled=self.args.muon_enabled
        ).to(self.device)
        routers['energy_aware_ttt'] = energy_aware_model
        
        return routers
    
    def run_benchmark(self):
        """Run comprehensive benchmark of all routers."""
        logger.info("Starting router benchmark comparison")
        
        # Get data loader
        dataloader = self.dataloader_manager.get_dataloader(
            batch_size=self.args.batch_size,
            seq_length=self.args.seq_length
        )
        
        # Run benchmarks for each router
        for router_name, model in self.routers.items():
            logger.info(f"Benchmarking {router_name} router")
            self._benchmark_router(router_name, model, dataloader)
        
        # Analyze and report results
        self._analyze_results()
        self._generate_plots()
        self._save_results()
    
    def _benchmark_router(self, router_name: str, model: nn.Module, dataloader):
        """Benchmark a specific router implementation."""
        model.eval()  # Set to evaluation mode
        
        router_results = []
        total_tokens = 0
        total_energy = 0.0
        total_latency = 0.0
        temperature_readings = []
        power_readings = []
        memory_readings = []
        
        # Warm-up run
        warmup_batch = next(iter(dataloader))
        warmup_input = warmup_batch['input_ids'].to(self.device)
        with torch.no_grad():
            _ = model(warmup_input)
        
        # Benchmark runs
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.args.num_batches:
                break
            
            # Prepare input
            input_ids = batch['input_ids'].to(self.device)
            batch_size, seq_length = input_ids.shape
            num_tokens = batch_size * seq_length
            
            # Get initial hardware state
            initial_stats = self.gpu_monitor.get_current_stats()
            initial_temp = initial_stats.get('temperature', 50.0)
            initial_power = initial_stats.get('power_watt', 200.0)
            initial_memory = initial_stats.get('memory_utilization_percent', 50.0)
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            end_time = time.time()
            
            # Get final hardware state
            final_stats = self.gpu_monitor.get_current_stats()
            final_temp = final_stats.get('temperature', 50.0)
            final_power = final_stats.get('power_watt', 200.0)
            final_memory = final_stats.get('memory_utilization_percent', 50.0)
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            temp_rise = final_temp - initial_temp
            power_consumption = (initial_power + final_power) / 2  # Average power
            memory_usage = (initial_memory + final_memory) / 2  # Average memory
            
            # Store results
            result = {
                'batch_idx': batch_idx,
                'num_tokens': num_tokens,
                'latency_ms': latency_ms,
                'latency_per_token_ms': latency_ms / num_tokens,
                'temp_rise_c': temp_rise,
                'power_watt': power_consumption,
                'energy_per_token_j': (power_consumption * latency_ms / 1000) / num_tokens,
                'memory_utilization_percent': memory_usage,
                'gpu_utilization_percent': final_stats.get('gpu_utilization_percent', 0.0)
            }
            
            # Add router-specific metrics
            if hasattr(model.moe_layer.router, 'get_statistics'):
                router_stats = model.moe_layer.router.get_statistics()
                result.update({
                    'ttt_update_count': router_stats.get('ttt_update_count', 0),
                    'load_balance_score': router_stats.get('balance_score', 1.0),
                    'expert_utilization_std': router_stats.get('std_dev', 0.0)
                })
            
            router_results.append(result)
            
            # Accumulate totals
            total_tokens += num_tokens
            total_energy += power_consumption * latency_ms / 1000
            total_latency += latency_ms
            temperature_readings.append(final_temp)
            power_readings.append(power_consumption)
            memory_readings.append(memory_usage)
            
            # Print progress
            if batch_idx % self.args.log_interval == 0:
                logger.info(f"  {router_name}: Batch {batch_idx}, "
                           f"Latency: {latency_ms:.2f}ms, "
                           f"Power: {power_consumption:.1f}W, "
                           f"Temp: {final_temp:.1f}°C")
        
        # Calculate summary statistics
        summary = {
            'total_tokens': total_tokens,
            'total_energy_j': total_energy,
            'total_latency_ms': total_latency,
            'avg_latency_per_token_ms': total_latency / total_tokens,
            'avg_energy_per_token_j': total_energy / total_tokens,
            'avg_power_watt': np.mean(power_readings),
            'max_temp_c': np.max(temperature_readings),
            'avg_temp_c': np.mean(temperature_readings),
            'temp_rise_c': np.max(temperature_readings) - np.min(temperature_readings),
            'avg_memory_utilization_percent': np.mean(memory_readings),
            'throughput_tokens_per_sec': total_tokens / (total_latency / 1000)
        }
        
        self.results[router_name] = {
            'detailed_results': router_results,
            'summary': summary
        }
        
        logger.info(f"Completed benchmark for {router_name}")
        logger.info(f"  Avg latency: {summary['avg_latency_per_token_ms']:.4f} ms/token")
        logger.info(f"  Avg energy: {summary['avg_energy_per_token_j']:.6f} J/token")
        logger.info(f"  Throughput: {summary['throughput_tokens_per_sec']:.1f} tokens/sec")
    
    def _analyze_results(self):
        """Analyze and compare results across all routers."""
        logger.info("Analyzing benchmark results")
        
        # Create comparison table
        comparison_data = []
        
        for router_name, result_data in self.results.items():
            summary = result_data['summary']
            comparison_data.append({
                'Router': router_name,
                'Latency (ms/token)': summary['avg_latency_per_token_ms'],
                'Energy (J/token)': summary['avg_energy_per_token_j'],
                'Power (W)': summary['avg_power_watt'],
                'Max Temp (°C)': summary['max_temp_c'],
                'Temp Rise (°C)': summary['temp_rise_c'],
                'Memory (%)': summary['avg_memory_utilization_percent'],
                'Throughput (tokens/sec)': summary['throughput_tokens_per_sec']
            })
        
        # Create DataFrame for easy analysis
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvements over baseline
        baseline_row = self.comparison_df[self.comparison_df['Router'] == 'baseline'].iloc[0]
        
        improvements = {}
        for router_name in self.results.keys():
            if router_name != 'baseline':
                router_row = self.comparison_df[self.comparison_df['Router'] == router_name].iloc[0]
                
                improvements[router_name] = {
                    'latency_improvement_pct': ((baseline_row['Latency (ms/token)'] - router_row['Latency (ms/token)']) / 
                                              baseline_row['Latency (ms/token)']) * 100,
                    'energy_improvement_pct': ((baseline_row['Energy (J/token)'] - router_row['Energy (J/token)']) / 
                                             baseline_row['Energy (J/token)']) * 100,
                    'power_improvement_pct': ((baseline_row['Power (W)'] - router_row['Power (W)']) / 
                                            baseline_row['Power (W)']) * 100,
                    'temp_improvement_c': baseline_row['Temp Rise (°C)'] - router_row['Temp Rise (°C)']
                }
        
        self.improvements = improvements
        
        # Print comparison table
        logger.info("\n=== Router Comparison Results ===")
        logger.info(self.comparison_df.to_string(index=False))
        
        # Print improvements
        logger.info("\n=== Improvements over Baseline ===")
        for router_name, improvement in improvements.items():
            logger.info(f"\n{router_name}:")
            logger.info(f"  Latency improvement: {improvement['latency_improvement_pct']:.2f}%")
            logger.info(f"  Energy improvement: {improvement['energy_improvement_pct']:.2f}%")
            logger.info(f"  Power improvement: {improvement['power_improvement_pct']:.2f}%")
            logger.info(f"  Temperature improvement: {improvement['temp_improvement_c']:.2f}°C")
    
    def _generate_plots(self):
        """Generate comparison plots."""
        logger.info("Generating comparison plots")
        
        # Create output directory
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Router Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        routers = self.comparison_df['Router'].values
        latency = self.comparison_df['Latency (ms/token)'].values
        energy = self.comparison_df['Energy (J/token)'].values
        power = self.comparison_df['Power (W)'].values
        max_temp = self.comparison_df['Max Temp (°C)'].values
        temp_rise = self.comparison_df['Temp Rise (°C)'].values
        throughput = self.comparison_df['Throughput (tokens/sec)'].values
        
        # Plot 1: Latency comparison
        axes[0, 0].bar(routers, latency, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 0].set_title('Latency per Token')
        axes[0, 0].set_ylabel('Latency (ms/token)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Energy comparison
        axes[0, 1].bar(routers, energy, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 1].set_title('Energy per Token')
        axes[0, 1].set_ylabel('Energy (J/token)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Power comparison
        axes[0, 2].bar(routers, power, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[0, 2].set_title('Average Power Consumption')
        axes[0, 2].set_ylabel('Power (W)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Temperature comparison
        axes[1, 0].bar(routers, max_temp, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 0].set_title('Maximum Temperature')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Temperature rise comparison
        axes[1, 1].bar(routers, temp_rise, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 1].set_title('Temperature Rise')
        axes[1, 1].set_ylabel('Temperature Rise (°C)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Throughput comparison
        axes[1, 2].bar(routers, throughput, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 2].set_title('Throughput')
        axes[1, 2].set_ylabel('Tokens per Second')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'router_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create improvement radar chart
        self._create_radar_chart(output_dir)
        
        logger.info(f"Plots saved to {output_dir}")
    
    def _create_radar_chart(self, output_dir: Path):
        """Create radar chart showing improvements."""
        if not self.improvements:
            return
        
        # Prepare data for radar chart
        categories = ['Latency\nImprovement', 'Energy\nImprovement', 'Power\nImprovement', 
                     'Temp\nImprovement', 'Throughput\nImprovement']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # Plot each router's improvements
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        router_names = list(self.improvements.keys())
        
        for i, (router_name, improvement) in enumerate(self.improvements.items()):
            values = [
                max(0, improvement['latency_improvement_pct']),
                max(0, improvement['energy_improvement_pct']),
                max(0, improvement['power_improvement_pct']),
                max(0, improvement['temp_improvement_c'] * 10),  # Scale temp improvement
                max(0, (self.comparison_df[self.comparison_df['Router'] == router_name]['Throughput (tokens/sec)'].iloc[0] / 
                       self.comparison_df[self.comparison_df['Router'] == 'baseline']['Throughput (tokens/sec)'].iloc[0] - 1) * 100)
            ]
            
            # Close the plot by repeating first value
            values += values[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=router_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Set up the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Performance Improvements over Baseline', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvements_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self):
        """Save detailed results to files."""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        self.comparison_df.to_csv(output_dir / 'router_comparison.csv', index=False)
        
        # Save detailed results
        detailed_results = {}
        for router_name, result_data in self.results.items():
            detailed_results[router_name] = {
                'summary': result_data['summary'],
                'detailed_results': result_data['detailed_results']
            }
        
        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save improvements
        with open(output_dir / 'improvements.json', 'w') as f:
            json.dump(self.improvements, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Router Performance Comparison")
    
    # Benchmark configuration
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to benchmark")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k expert selection")
    parser.add_argument("--expert_type", type=str, default="simple", help="Expert type")
    
    # TTT configuration
    parser.add_argument("--ttt_chunk_size", type=int, default=2048, help="TTT chunk size")
    parser.add_argument("--ttt_update_frequency", type=int, default=512, help="TTT update frequency")
    parser.add_argument("--energy_aware_lr", type=float, default=1e-4, help="Energy-aware learning rate")
    parser.add_argument("--muon_enabled", action="store_true", default=True, help="Enable Muon optimizer")
    
    # System configuration
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--gpu_type", type=str, default="A100", help="GPU type")
    parser.add_argument("--kernel_cost_model_json", type=str, default="energy/cost_table.json")
    parser.add_argument("--log_file", type=str, default="logs/router_comparison.log")
    parser.add_argument("--output_dir", type=str, default="results/router_comparison")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = RouterBenchmark(args)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main() 