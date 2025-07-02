#!/usr/bin/env python3
"""
HPC Single GPU TTT Test with Kernel Cost Model Integration

This script is designed to run on HPC clusters with single GPU allocation.
It tests the EnergyAwareTTTRouter vs regular TTT on one GPU with hardware monitoring
and comprehensive kernel cost model testing.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging for HPC environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hpc_ttt_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Check if we're on a cluster and set device accordingly
def setup_cluster_environment():
    """Setup environment for HPC cluster."""
    # Check for SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        logger.info(f"Running on SLURM cluster, Job ID: {os.environ['SLURM_JOB_ID']}")
        logger.info(f"SLURM_NODEID: {os.environ.get('SLURM_NODEID', 'N/A')}")
        logger.info(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'N/A')}")
        
        # Set device based on SLURM GPU allocation
        if 'SLURM_CUDA_DEVICES' in os.environ:
            gpu_id = int(os.environ['SLURM_CUDA_DEVICES'].split(',')[0])
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Using GPU {gpu_id} from SLURM allocation")
        else:
            device = torch.device('cuda:0')
            logger.info("Using default GPU 0")
    
    # Check for other cluster environments
    elif 'PBS_JOBID' in os.environ:
        logger.info(f"Running on PBS cluster, Job ID: {os.environ['PBS_JOBID']}")
        device = torch.device('cuda:0')
    elif 'LSB_JOBID' in os.environ:
        logger.info(f"Running on LSF cluster, Job ID: {os.environ['LSB_JOBID']}")
        device = torch.device('cuda:0')
    else:
        logger.info("Running on local machine")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return device


class SimpleTransformer(nn.Module):
    """Simple transformer for HPC testing."""
    
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


class MockGpuMonitor:
    """Mock GPU monitor for HPC environments where nvidia-smi might not be available."""
    
    def __init__(self, device):
        self.device = device
        self.mock_stats = {
            'temperature': 65.0,
            'power_watt': 200.0,
            'memory_utilization_percent': 70.0,
            'gpu_utilization_percent': 80.0,
            'thermal_state': 'warm'
        }
    
    def get_current_stats(self):
        """Get current GPU stats (mock implementation for HPC)."""
        if torch.cuda.is_available():
            # Try to get real stats if possible
            try:
                # Get memory usage
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9   # GB
                
                # Estimate power based on memory usage
                estimated_power = 50.0 + (memory_allocated * 100.0)
                
                # Estimate temperature based on utilization
                gpu_util = min(100.0, memory_allocated * 200.0)  # Rough estimate
                estimated_temp = 50.0 + (gpu_util * 0.3)
                
                return {
                    'temperature': estimated_temp,
                    'power_watt': estimated_power,
                    'memory_utilization_percent': (memory_allocated / 24.0) * 100,  # Assume 24GB GPU
                    'gpu_utilization_percent': gpu_util,
                    'thermal_state': 'warm' if estimated_temp > 70 else 'cool'
                }
            except:
                return self.mock_stats.copy()
        else:
            return self.mock_stats.copy()


def test_kernel_cost_model_integration(kernel_cost_model, device):
    """Test kernel cost model functionality and integration."""
    logger.info("\n🔧 Testing Kernel Cost Model Integration")
    logger.info("=" * 50)
    
    # Test different operation types
    test_ops = ["ffn_gate", "ffn_up", "ffn_down", "attention_qk", "attention_av", "moe_router"]
    test_batch_sizes = [1, 8, 32, 128]
    
    logger.info("Testing operation costs across batch sizes:")
    for op in test_ops:
        logger.info(f"\n  {op}:")
        for bs in test_batch_sizes:
            cost = kernel_cost_model.get_cost(op, bs)
            logger.info(f"    batch_size={bs:3d}: energy={cost['energy_joules']:.6f}J, "
                       f"latency={cost['latency_ms']:.3f}ms, temp_impact={cost['temp_impact']:.3f}°C")
    
    # Test thermal throttling effects
    logger.info("\nTesting thermal throttling effects:")
    op = "ffn_gate"
    bs = 32
    normal_temp = 50.0
    hot_temp = 85.0
    
    normal_cost = kernel_cost_model.get_cost(op, bs, current_temp=normal_temp)
    hot_cost = kernel_cost_model.get_cost(op, bs, current_temp=hot_temp)
    
    latency_increase = (hot_cost['latency_ms'] / normal_cost['latency_ms'] - 1) * 100
    energy_increase = (hot_cost['energy_joules'] / normal_cost['energy_joules'] - 1) * 100
    
    logger.info(f"  {op} (batch_size={bs}):")
    logger.info(f"    Normal temp ({normal_temp}°C): {normal_cost['latency_ms']:.3f}ms, {normal_cost['energy_joules']:.6f}J")
    logger.info(f"    Hot temp ({hot_temp}°C): {hot_cost['latency_ms']:.3f}ms, {hot_cost['energy_joules']:.6f}J")
    logger.info(f"    Thermal impact: +{latency_increase:.1f}% latency, +{energy_increase:.1f}% energy")
    
    # Test memory pressure effects
    logger.info("\nTesting memory pressure effects:")
    low_memory = 0.3
    high_memory = 0.9
    
    low_mem_cost = kernel_cost_model.get_cost(op, bs, memory_pressure=low_memory)
    high_mem_cost = kernel_cost_model.get_cost(op, bs, memory_pressure=high_memory)
    
    mem_latency_increase = (high_mem_cost['latency_ms'] / low_mem_cost['latency_ms'] - 1) * 100
    mem_energy_increase = (high_mem_cost['energy_joules'] / low_mem_cost['energy_joules'] - 1) * 100
    
    logger.info(f"  {op} (batch_size={bs}):")
    logger.info(f"    Low memory pressure ({low_memory*100:.0f}%): {low_mem_cost['latency_ms']:.3f}ms, {low_mem_cost['energy_joules']:.6f}J")
    logger.info(f"    High memory pressure ({high_memory*100:.0f}%): {high_mem_cost['latency_ms']:.3f}ms, {high_mem_cost['energy_joules']:.6f}J")
    logger.info(f"    Memory impact: +{mem_latency_increase:.1f}% latency, +{mem_energy_increase:.1f}% energy")
    
    # Test thermal-safe batch size recommendation
    logger.info("\nTesting thermal-safe batch size recommendations:")
    for op in ["ffn_gate", "attention_qk", "moe_router"]:
        safe_batch = kernel_cost_model.get_thermal_safe_batch_size(op, current_temp=80.0, max_temp_increase=3.0)
        logger.info(f"  {op}: thermal-safe batch size at 80°C = {safe_batch}")
    
    return {
        'test_ops': test_ops,
        'thermal_throttling_factor': latency_increase,
        'memory_pressure_factor': mem_latency_increase,
        'thermal_safe_batch_sizes': {
            op: kernel_cost_model.get_thermal_safe_batch_size(op, current_temp=80.0, max_temp_increase=3.0)
            for op in test_ops[:3]
        }
    }


def measure_performance_hpc(model, input_ids, model_name, use_ttt=False, num_runs=10, gpu_monitor=None, kernel_cost_model=None):
    """Measure performance metrics optimized for HPC environment with kernel cost model integration."""
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    num_tokens = batch_size * seq_len
    
    # Warm up
    logger.info(f"Warming up {model_name}...")
    with torch.no_grad():
        for _ in range(3):
            model(input_ids, use_ttt=use_ttt)
    
    # Synchronize before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure performance
    latencies = []
    memory_usage = []
    power_readings = []
    temp_readings = []
    predicted_energy_costs = []
    
    logger.info(f"Running {num_runs} performance measurements for {model_name}...")
    
    for run in range(num_runs):
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get initial stats
        if gpu_monitor:
            initial_stats = gpu_monitor.get_current_stats()
            initial_power = initial_stats.get('power_watt', 200.0)
            initial_temp = initial_stats.get('temperature', 65.0)
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(device)
        
        # Start timing
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, use_ttt=use_ttt)
        
        # Synchronize and measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(device)
            memory_used_gb = (memory_after - memory_before) / 1e9
        else:
            memory_used_gb = 0.0
        
        # Get final stats
        if gpu_monitor:
            final_stats = gpu_monitor.get_current_stats()
            final_power = final_stats.get('power_watt', 200.0)
            final_temp = final_stats.get('temperature', 65.0)
            
            # Use average power
            avg_power = (initial_power + final_power) / 2
            avg_temp = (initial_temp + final_temp) / 2
        else:
            avg_power = 200.0
            avg_temp = 65.0
        
        # Calculate energy
        energy_joules = (avg_power * latency_ms) / 1000
        
        # Predict energy using kernel cost model if available
        predicted_energy = 0.0
        if kernel_cost_model and hasattr(model, 'config'):
            try:
                # Estimate operations based on model type
                if hasattr(model, 'router'):  # MoE model
                    expert_ops = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu", "moe_router"]
                    if hasattr(model.config, 'expert_type') and model.config.expert_type == "quantized":
                        expert_ops.extend(["quantize_w8a16", "dequantize_w8a16"])
                    
                    # Get current hardware state
                    gpu_stats = gpu_monitor.get_current_stats() if gpu_monitor else {}
                    current_temp = gpu_stats.get('temperature', 50.0)
                    memory_pressure = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0
                    
                    # Calculate predicted energy for expert operations
                    effective_batch = batch_size / getattr(model.config, 'top_k', 2)
                    for op in expert_ops:
                        op_cost = kernel_cost_model.get_cost(op, int(effective_batch), 
                                                           current_temp=current_temp, 
                                                           memory_pressure=memory_pressure)
                        predicted_energy += op_cost.get('energy_joules', 0.0)
                else:  # Simple transformer
                    transformer_ops = ["attention_qk", "attention_av", "attention_proj", "ffn_gate", "ffn_up", "ffn_down"]
                    for op in transformer_ops:
                        op_cost = kernel_cost_model.get_cost(op, batch_size)
                        predicted_energy += op_cost.get('energy_joules', 0.0)
            except Exception as e:
                logger.warning(f"Error predicting energy with kernel cost model: {e}")
        
        # Store measurements
        latencies.append(latency_ms)
        memory_usage.append(memory_used_gb)
        power_readings.append(avg_power)
        temp_readings.append(avg_temp)
        predicted_energy_costs.append(predicted_energy)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_memory = np.mean(memory_usage)
    avg_power = np.mean(power_readings)
    avg_temperature = np.mean(temp_readings)
    avg_predicted_energy = np.mean(predicted_energy_costs)
    
    # Calculate throughput
    throughput = (num_tokens * 1000) / avg_latency  # tokens per second
    
    # Calculate energy per token
    energy_per_token = energy_joules / num_tokens
    
    return {
        'model_name': model_name,
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'avg_memory_gb': avg_memory,
        'avg_power_w': avg_power,
        'avg_temperature_c': avg_temperature,
        'energy_per_token_j': energy_per_token,
        'throughput_tokens_per_sec': throughput,
        'ttt_update_count': model.ttt_update_count if hasattr(model, 'ttt_update_count') else 0,
        'num_runs': num_runs,
        'predicted_energy_j': avg_predicted_energy,
        'energy_prediction_accuracy': abs(avg_predicted_energy - energy_joules) / energy_joules if energy_joules > 0 else 0
    }


def run_hpc_comparison(args):
    """Run the HPC comparison test with kernel cost model integration."""
    logger.info("🚀 Starting HPC Single GPU TTT Comparison with Kernel Cost Model")
    logger.info("=" * 70)
    
    # Setup cluster environment
    device = setup_cluster_environment()
    logger.info(f"Using device: {device}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Create GPU monitor
    gpu_monitor = MockGpuMonitor(device)
    
    # Create kernel cost model
    logger.info("\n🔧 Initializing Kernel Cost Model...")
    try:
        from src.kernelcostmodel import KernelCostModel
        kernel_cost_model = KernelCostModel(gpu_type="A100")  # Adjust based on your GPU
        logger.info("✅ Kernel Cost Model initialized successfully")
        
        # Test kernel cost model integration
        kcm_test_results = test_kernel_cost_model_integration(kernel_cost_model, device)
        
    except ImportError as e:
        logger.warning(f"KernelCostModel not available: {e}")
        kernel_cost_model = None
        kcm_test_results = None
    
    # Model parameters
    d_model = args.d_model
    num_layers = args.num_layers
    batch_size = args.batch_size
    seq_length = args.seq_length
    num_runs = args.num_runs
    
    # Generate test data
    input_ids = torch.randint(0, 10000, (batch_size, seq_length)).to(device)
    logger.info(f"Test data: {batch_size} batches, {seq_length} sequence length")
    
    results = []
    
    # Test 1: Simple Transformer (No TTT)
    logger.info("\n📊 Testing Simple Transformer (No TTT)...")
    simple_model = SimpleTransformer(d_model=d_model, num_layers=num_layers).to(device)
    simple_results = measure_performance_hpc(
        simple_model, input_ids, "Simple Transformer", 
        use_ttt=False, num_runs=num_runs, gpu_monitor=gpu_monitor, 
        kernel_cost_model=kernel_cost_model
    )
    results.append(simple_results)
    
    # Test 2: Simple Transformer with TTT
    logger.info("\n📊 Testing Simple Transformer with TTT...")
    ttt_model = SimpleTransformer(d_model=d_model, num_layers=num_layers).to(device)
    ttt_results = measure_performance_hpc(
        ttt_model, input_ids, "Simple Transformer + TTT", 
        use_ttt=True, num_runs=num_runs, gpu_monitor=gpu_monitor,
        kernel_cost_model=kernel_cost_model
    )
    results.append(ttt_results)
    
    # Test 3: MoE with EnergyAwareTTTRouter (if available)
    if args.test_moe:
        logger.info("\n📊 Testing MoE with EnergyAwareTTTRouter...")
        try:
            # Import MoE components
            from src.moe_models import MoEConfig, MoETransformerBlock
            from src.routers import EnergyAwareTTTRouter
            from src.monitor import GpuSystemMonitor
            
            # Create MoE config
            moe_config = MoEConfig(
                d_model=d_model,
                num_experts=args.num_experts,
                top_k=args.top_k,
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
            gpu_system_monitor = GpuSystemMonitor(device_id=device.index if device.type == 'cuda' else 0)
            
            # Create MoE model (only if kernel cost model is available)
            if kernel_cost_model is not None:
                moe_model = MoETransformerBlock(
                    moe_config,
                    kernel_cost_model,
                    gpu_system_monitor
                ).to(device)
                
                # Replace router with EnergyAwareTTTRouter
                moe_model.router = EnergyAwareTTTRouter(
                    config=moe_config,
                    kernel_cost_model=kernel_cost_model,
                    gpu_system_monitor=gpu_system_monitor,
                    ttt_chunk_size=args.ttt_chunk_size,
                    ttt_update_frequency=args.ttt_update_frequency,
                    energy_aware_lr=args.energy_aware_lr,
                    muon_enabled=args.muon_enabled
                ).to(device)
            else:
                logger.warning("Kernel cost model not available, skipping MoE test")
            # Test MoE model
            moe_results = measure_performance_hpc(
                moe_model, input_ids, "MoE + EnergyAwareTTTRouter", 
                use_ttt=False, num_runs=num_runs, gpu_monitor=gpu_monitor,
                kernel_cost_model=kernel_cost_model
            )
            results.append(moe_results)
            
        except ImportError as e:
            logger.warning(f"MoE components not available: {e}")
            logger.info("Skipping MoE test")
        except Exception as e:
            logger.error(f"Error testing MoE: {e}")
            logger.info("Skipping MoE test")
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("📈 RESULTS COMPARISON")
    logger.info("=" * 70)
    
    for result in results:
        logger.info(f"\n{result['model_name']}:")
        logger.info(f"  Latency: {result['avg_latency_ms']:.2f} ± {result['std_latency_ms']:.2f} ms")
        logger.info(f"  Memory: {result['avg_memory_gb']:.3f} GB")
        logger.info(f"  Power: {result['avg_power_w']:.1f} W")
        logger.info(f"  Temperature: {result['avg_temperature_c']:.1f}°C")
        logger.info(f"  Energy per token: {result['energy_per_token_j']:.6f} J")
        logger.info(f"  Throughput: {result['throughput_tokens_per_sec']:.1f} tokens/sec")
        if result['ttt_update_count'] > 0:
            logger.info(f"  TTT updates: {result['ttt_update_count']}")
        if kernel_cost_model and result['predicted_energy_j'] > 0:
            logger.info(f"  Predicted energy: {result['predicted_energy_j']:.6f} J")
            logger.info(f"  Energy prediction accuracy: {result['energy_prediction_accuracy']*100:.1f}%")
    
    # Calculate improvements
    if len(results) >= 2:
        logger.info("\n" + "=" * 70)
        logger.info("🎯 IMPROVEMENTS OVER BASELINE")
        logger.info("=" * 70)
        
        baseline = results[0]
        
        for result in results[1:]:
            latency_improvement = ((baseline['avg_latency_ms'] - result['avg_latency_ms']) / baseline['avg_latency_ms']) * 100
            power_improvement = ((baseline['avg_power_w'] - result['avg_power_w']) / baseline['avg_power_w']) * 100
            energy_improvement = ((baseline['energy_per_token_j'] - result['energy_per_token_j']) / baseline['energy_per_token_j']) * 100
            throughput_improvement = ((result['throughput_tokens_per_sec'] - baseline['throughput_tokens_per_sec']) / baseline['throughput_tokens_per_sec']) * 100
            
            logger.info(f"\n{result['model_name']} vs Baseline:")
            logger.info(f"  Latency: {latency_improvement:+.1f}%")
            logger.info(f"  Power: {power_improvement:+.1f}%")
            logger.info(f"  Energy per token: {energy_improvement:+.1f}%")
            logger.info(f"  Throughput: {throughput_improvement:+.1f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_dir / "hpc_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save kernel cost model test results
    if kcm_test_results:
        with open(output_dir / "kernel_cost_model_test_results.json", 'w') as f:
            json.dump(kcm_test_results, f, indent=2, default=str)
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU',
        'gpu_memory_gb': torch.cuda.get_device_properties(device).total_memory / 1e9 if torch.cuda.is_available() else 0,
        'test_parameters': {
            'd_model': d_model,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'num_runs': num_runs
        },
        'results': results,
        'kernel_cost_model_available': kernel_cost_model is not None,
        'kernel_cost_model_test_results': kcm_test_results
    }
    
    with open(output_dir / "hpc_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to {output_dir}")
    logger.info("✅ HPC test completed!")


def main():
    parser = argparse.ArgumentParser(description="HPC Single GPU TTT Comparison with Kernel Cost Model")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of test runs")
    
    # MoE parameters
    parser.add_argument("--test_moe", action="store_true", help="Test MoE with EnergyAwareTTTRouter")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k expert selection")
    
    # TTT parameters
    parser.add_argument("--ttt_chunk_size", type=int, default=512, help="TTT chunk size")
    parser.add_argument("--ttt_update_frequency", type=int, default=128, help="TTT update frequency")
    parser.add_argument("--energy_aware_lr", type=float, default=1e-4, help="Energy-aware learning rate")
    parser.add_argument("--muon_enabled", action="store_true", help="Enable Muon TTT")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/hpc_test", help="Output directory")
    
    args = parser.parse_args()
    
    # Run the comparison
    run_hpc_comparison(args)


if __name__ == "__main__":
    main() 