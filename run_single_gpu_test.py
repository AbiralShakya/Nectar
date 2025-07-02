#!/usr/bin/env python3
"""
Simple Single GPU TTT Test

This script runs a quick comparison between regular TTT and EnergyAwareTTTRouter
on a single GPU with a small transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path

# Import our modules
from src.moe_models import MoEConfig
from src.routers import EnergyAwareTTTRouter
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor


class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""
    
    def __init__(self, d_model=256, num_layers=4, vocab_size=10000):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # TTT fast weights
        self.ttt_fast_weights = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.ttt_optimizer = torch.optim.AdamW([self.ttt_fast_weights], lr=1e-4)
        self.ttt_buffer = []
        self.ttt_update_count = 0
    
    def forward(self, input_ids, use_ttt=False):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Apply TTT if enabled
        if use_ttt:
            hidden_states = hidden_states + torch.tanh(hidden_states @ self.ttt_fast_weights)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Output
        logits = self.output_projection(hidden_states)
        
        return {'logits': logits, 'hidden_states': hidden_states}


def measure_performance(model, input_ids, model_name, use_ttt=False, num_runs=10):
    """Measure performance metrics for a model."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    num_tokens = batch_size * seq_len
    
    # Warm up
    with torch.no_grad():
        for _ in range(3):
            model(input_ids, use_ttt=use_ttt)
    
    # Measure performance
    latencies = []
    memory_usage = []
    
    for _ in range(num_runs):
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids, use_ttt=use_ttt)
        end_time = time.time()
        
        # Measure memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
            memory_usage.append((memory_after - memory_before) / 1e9)  # GB
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Calculate metrics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_memory = np.mean(memory_usage) if memory_usage else 0.0
    
    # Estimate power (rough approximation)
    if torch.cuda.is_available():
        # Rough power estimation based on memory usage and computation
        estimated_power = 50.0 + (avg_memory * 100.0) + (avg_latency * 0.1)
    else:
        estimated_power = 0.0
    
    # Calculate energy per token
    energy_per_token = (estimated_power * avg_latency / 1000) / num_tokens
    throughput = num_tokens / (avg_latency / 1000)
    
    return {
        'model_name': model_name,
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'avg_memory_gb': avg_memory,
        'estimated_power_w': estimated_power,
        'energy_per_token_j': energy_per_token,
        'throughput_tokens_per_sec': throughput,
        'ttt_update_count': model.ttt_update_count if hasattr(model, 'ttt_update_count') else 0
    }


def main():
    """Run the single GPU TTT comparison."""
    print("🚀 Single GPU TTT Comparison")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    d_model = 256
    num_layers = 4
    batch_size = 8
    seq_length = 512
    num_runs = 20
    
    # Generate test data
    input_ids = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
    print(f"Test data: {batch_size} batches, {seq_length} sequence length")
    
    # Test 1: Simple Transformer (No TTT)
    print("\n📊 Testing Simple Transformer (No TTT)...")
    simple_model = SimpleTransformer(d_model=d_model, num_layers=num_layers).to(device)
    simple_results = measure_performance(simple_model, input_ids, "Simple Transformer", use_ttt=False, num_runs=num_runs)
    
    # Test 2: Simple Transformer with TTT
    print("📊 Testing Simple Transformer with TTT...")
    ttt_model = SimpleTransformer(d_model=d_model, num_layers=num_layers).to(device)
    ttt_results = measure_performance(ttt_model, input_ids, "Simple Transformer + TTT", use_ttt=True, num_runs=num_runs)
    
    # Test 3: MoE with EnergyAwareTTTRouter
    print("📊 Testing MoE with EnergyAwareTTTRouter...")
    
    # Create MoE config
    moe_config = MoEConfig(
        d_model=d_model,
        num_experts=4,
        top_k=2,
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
    
    # Create components
    kernel_cost_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor(device_id=0)
    
    # Create MoE model with EnergyAwareTTTRouter
    from src.moe_models import MoETransformerBlock
    
    moe_model = MoETransformerBlock(
        moe_config,
        kernel_cost_model,
        gpu_monitor
    ).to(device)
    
    # Replace router with EnergyAwareTTTRouter
    moe_model.router = EnergyAwareTTTRouter(
        config=moe_config,
        kernel_cost_model=kernel_cost_model,
        gpu_system_monitor=gpu_monitor,
        ttt_chunk_size=512,
        ttt_update_frequency=128,
        energy_aware_lr=1e-4,
        muon_enabled=True
    ).to(device)
    
    # Test MoE model
    moe_results = measure_performance(moe_model, input_ids, "MoE + EnergyAwareTTTRouter", use_ttt=False, num_runs=num_runs)
    
    # Print results
    print("\n" + "=" * 50)
    print("📈 RESULTS COMPARISON")
    print("=" * 50)
    
    results = [simple_results, ttt_results, moe_results]
    
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  Latency: {result['avg_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")
        print(f"  Memory: {result['avg_memory_gb']:.3f} GB")
        print(f"  Power: {result['estimated_power_w']:.1f} W")
        print(f"  Energy per token: {result['energy_per_token_j']:.6f} J")
        print(f"  Throughput: {result['throughput_tokens_per_sec']:.1f} tokens/sec")
        if result['ttt_update_count'] > 0:
            print(f"  TTT updates: {result['ttt_update_count']}")
    
    # Calculate improvements
    print("\n" + "=" * 50)
    print("🎯 IMPROVEMENTS OVER BASELINE")
    print("=" * 50)
    
    baseline = simple_results
    
    for result in [ttt_results, moe_results]:
        if result == baseline:
            continue
            
        latency_improvement = ((baseline['avg_latency_ms'] - result['avg_latency_ms']) / baseline['avg_latency_ms']) * 100
        power_improvement = ((baseline['estimated_power_w'] - result['estimated_power_w']) / baseline['estimated_power_w']) * 100
        energy_improvement = ((baseline['energy_per_token_j'] - result['energy_per_token_j']) / baseline['energy_per_token_j']) * 100
        throughput_improvement = ((result['throughput_tokens_per_sec'] - baseline['throughput_tokens_per_sec']) / baseline['throughput_tokens_per_sec']) * 100
        
        print(f"\n{result['model_name']} vs Baseline:")
        print(f"  Latency: {latency_improvement:+.1f}%")
        print(f"  Power: {power_improvement:+.1f}%")
        print(f"  Energy per token: {energy_improvement:+.1f}%")
        print(f"  Throughput: {throughput_improvement:+.1f}%")
    
    # Save results
    output_dir = Path("results/single_gpu_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for easy saving
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "comparison_results.csv", index=False)
    
    print(f"\n💾 Results saved to {output_dir}")
    
    # Create simple plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Single GPU TTT Comparison', fontsize=16)
        
        metrics = ['avg_latency_ms', 'estimated_power_w', 'energy_per_token_j', 'throughput_tokens_per_sec']
        metric_names = ['Latency (ms)', 'Power (W)', 'Energy (J/token)', 'Throughput (tokens/sec)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            row, col = i // 2, i % 2
            
            values = [result[metric] for result in results]
            names = [result['model_name'] for result in results]
            
            axes[row, col].bar(names, values)
            axes[row, col].set_title(metric_name)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved to {output_dir / 'comparison_plot.png'}")
        
    except ImportError:
        print("📊 Matplotlib not available, skipping plot generation")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main() 