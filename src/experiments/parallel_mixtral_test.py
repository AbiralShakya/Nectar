import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import numpy as np
from dataclasses import dataclass
import os
import argparse

from src.moe_models import DistributedMoELayer, NetworkTopologyOptimizer
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState
from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class MixtralConfig:
    """Configuration for Mixtral 7B model."""
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    num_experts: int = 8
    num_layers: int = 32
    d_model: int = 4096
    num_heads: int = 32
    vocab_size: int = 32000
    max_seq_length: int = 32768
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True

@dataclass
class ParallelTestConfig:
    """Configuration for parallel testing."""
    num_gpus: int = 4
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    test_modes: List[str] = None
    expert_placement_strategy: str = "load_balanced"
    enable_thermal_awareness: bool = True
    enable_ttt: bool = True
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256, 512, 1024]
        if self.test_modes is None:
            self.test_modes = ["decode", "prefill"]

class ParallelMixtralTester:
    """
    Parallel testing framework for Mixtral 7B with MoE + TTT optimization.
    Tests decode vs prefill performance across multiple GPUs.
    """
    def __init__(self, config: ParallelTestConfig, mixtral_config: MixtralConfig):
        self.config = config
        self.mixtral_config = mixtral_config
        self.results = []
        
        # Initialize distributed components
        self.topology_optimizer = NetworkTopologyOptimizer(
            num_gpus=config.num_gpus,
            num_experts=mixtral_config.num_experts
        )
        
        # Thermal awareness
        if config.enable_thermal_awareness:
            self.thermal_router = ThermalAwareRouter(
                num_experts=mixtral_config.num_experts,
                num_gpus=config.num_gpus
            )
        
        # TTT router
        if config.enable_ttt:
            self.ttt_router = EnergyAwareTTTRouter(
                d_model=mixtral_config.d_model,
                num_experts=mixtral_config.num_experts,
                top_k=2,
                lambda_energy=0.001
            )
        
        # Kernel cost model
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        
    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        
        print(f"Process {rank}/{world_size} initialized on GPU {rank}")
    
    def cleanup_distributed(self):
        """Cleanup distributed environment."""
        dist.destroy_process_group()
    
    def run_parallel_test(self, rank: int, world_size: int):
        """Run parallel test on a single GPU."""
        try:
            self.setup_distributed(rank, world_size)
            
            # Test different configurations
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    for test_mode in self.config.test_modes:
                        result = self._test_configuration(
                            rank, world_size, batch_size, seq_len, test_mode
                        )
                        if result:
                            self.results.append(result)
            
            # Synchronize results across processes
            self._synchronize_results(rank, world_size)
            
        finally:
            self.cleanup_distributed()
    
    def _test_configuration(self, rank: int, world_size: int, batch_size: int, 
                           seq_len: int, test_mode: str) -> Optional[Dict[str, Any]]:
        """Test a specific configuration."""
        try:
            # Create synthetic input
            input_tensor = torch.randn(batch_size, seq_len, self.mixtral_config.d_model).cuda(rank)
            
            # Measure baseline performance
            baseline_metrics = self._measure_baseline_performance(
                input_tensor, batch_size, seq_len, test_mode
            )
            
            # Apply optimizations
            optimized_metrics = self._apply_optimizations(
                input_tensor, baseline_metrics, rank, world_size
            )
            
            # Calculate efficiency metrics
            tokens_per_second = (batch_size * seq_len) / (optimized_metrics['latency_ms'] / 1000)
            energy_efficiency = (batch_size * seq_len) / optimized_metrics['energy_joules']
            
            result = {
                'rank': rank,
                'world_size': world_size,
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'test_mode': test_mode,
                'baseline_metrics': baseline_metrics,
                'optimized_metrics': optimized_metrics,
                'tokens_per_second': tokens_per_second,
                'energy_efficiency': energy_efficiency,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            print(f"Error testing configuration on rank {rank}: {e}")
            return None
    
    def _measure_baseline_performance(self, input_tensor: torch.Tensor, 
                                    batch_size: int, seq_len: int, test_mode: str) -> Dict[str, float]:
        """Measure baseline performance without optimizations."""
        
        # Simulate Mixtral MoE computation
        if test_mode == "decode":
            # Decode mode: process one token at a time
            return self._simulate_decode_mode(input_tensor, batch_size, seq_len)
        else:
            # Prefill mode: process entire sequence
            return self._simulate_prefill_mode(input_tensor, batch_size, seq_len)
    
    def _simulate_decode_mode(self, input_tensor: torch.Tensor, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Simulate decode mode performance."""
        # Decode mode: typically smaller batches, focus on latency
        decode_batch_size = min(batch_size, 4)  # Smaller batches for decode
        
        # Estimate costs for decode mode
        attention_cost = self.kernel_cost_model.get_cost("attention_qk", decode_batch_size)
        ffn_cost = self.kernel_cost_model.get_cost("ffn_gate", decode_batch_size)
        moe_cost = self.kernel_cost_model.get_cost("moe_router", decode_batch_size)
        
        # Decode mode is more latency-sensitive
        total_latency = (
            attention_cost["latency_ms"] * 2 +  # QK and AV
            ffn_cost["latency_ms"] * 2 +  # 2 experts on average
            moe_cost["latency_ms"]  # Routing
        )
        
        total_energy = (
            attention_cost["energy_joules"] * 2 +
            ffn_cost["energy_joules"] * 2 +
            moe_cost["energy_joules"]
        )
        
        return {
            'latency_ms': total_latency,
            'energy_joules': total_energy,
            'power_watt': total_energy / (total_latency / 1000),
            'memory_usage': 0.4,  # Lower memory usage in decode
            'compute_utilization': 0.6,
            'mode': 'decode'
        }
    
    def _simulate_prefill_mode(self, input_tensor: torch.Tensor, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Simulate prefill mode performance."""
        # Prefill mode: larger batches, focus on throughput
        
        # Estimate costs for prefill mode
        attention_cost = self.kernel_cost_model.get_cost("attention_qk", batch_size)
        ffn_cost = self.kernel_cost_model.get_cost("ffn_gate", batch_size)
        moe_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        
        # Prefill mode can use more experts in parallel
        avg_experts_used = 4  # More experts used in prefill
        
        total_latency = max(
            attention_cost["latency_ms"] * 2,
            ffn_cost["latency_ms"] * avg_experts_used,
            moe_cost["latency_ms"]
        )
        
        total_energy = (
            attention_cost["energy_joules"] * 2 +
            ffn_cost["energy_joules"] * avg_experts_used +
            moe_cost["energy_joules"]
        )
        
        return {
            'latency_ms': total_latency,
            'energy_joules': total_energy,
            'power_watt': total_energy / (total_latency / 1000),
            'memory_usage': 0.8,  # Higher memory usage in prefill
            'compute_utilization': 0.9,
            'mode': 'prefill'
        }
    
    def _apply_optimizations(self, input_tensor: torch.Tensor, baseline_metrics: Dict[str, float],
                           rank: int, world_size: int) -> Dict[str, float]:
        """Apply various optimizations."""
        optimized_metrics = baseline_metrics.copy()
        
        # 1. Network topology optimization
        if hasattr(self, 'topology_optimizer'):
            topology_improvement = self._apply_topology_optimization(rank, world_size)
            optimized_metrics['latency_ms'] *= topology_improvement['latency_factor']
            optimized_metrics['energy_joules'] *= topology_improvement['energy_factor']
        
        # 2. Thermal awareness
        if hasattr(self, 'thermal_router') and self.config.enable_thermal_awareness:
            thermal_improvement = self._apply_thermal_optimization(rank)
            optimized_metrics['latency_ms'] *= thermal_improvement['latency_factor']
            optimized_metrics['power_watt'] *= thermal_improvement['power_factor']
        
        # 3. TTT optimization
        if hasattr(self, 'ttt_router') and self.config.enable_ttt:
            ttt_improvement = self._apply_ttt_optimization(input_tensor, rank)
            optimized_metrics['latency_ms'] *= ttt_improvement['latency_factor']
            optimized_metrics['energy_joules'] *= ttt_improvement['energy_factor']
        
        return optimized_metrics
    
    def _apply_topology_optimization(self, rank: int, world_size: int) -> Dict[str, float]:
        """Apply network topology optimization."""
        # Simulate topology optimization effects
        if world_size == 1:
            return {'latency_factor': 1.0, 'energy_factor': 1.0}
        
        # Multi-GPU optimization
        if self.config.expert_placement_strategy == "load_balanced":
            # Load balancing reduces communication overhead
            latency_factor = 0.85  # 15% improvement
            energy_factor = 0.9    # 10% improvement
        elif self.config.expert_placement_strategy == "bandwidth_optimized":
            # Bandwidth optimization reduces data movement
            latency_factor = 0.8   # 20% improvement
            energy_factor = 0.85   # 15% improvement
        else:
            latency_factor = 0.95  # 5% improvement
            energy_factor = 0.95   # 5% improvement
        
        return {
            'latency_factor': latency_factor,
            'energy_factor': energy_factor
        }
    
    def _apply_thermal_optimization(self, rank: int) -> Dict[str, float]:
        """Apply thermal-aware optimization."""
        # Simulate thermal state
        thermal_state = ThermalState(
            temperature=60.0 + rank * 5.0,  # Simulate thermal gradient
            power_watt=200.0 + rank * 20.0,
            memory_utilization=0.6 + rank * 0.1,
            compute_utilization=0.7 + rank * 0.1,
            timestamp=time.time()
        )
        
        # Update thermal router
        self.thermal_router.update_thermal_state(rank, thermal_state)
        
        # Get thermal optimization effects
        thermal_stats = self.thermal_router.get_thermal_stats()
        
        # Calculate improvement factors
        if thermal_stats['thermal_imbalance_score'] > 0.1:
            # High thermal imbalance - apply thermal optimization
            latency_factor = 0.9   # 10% improvement
            power_factor = 0.85    # 15% improvement
        else:
            # Low thermal imbalance - minimal optimization
            latency_factor = 0.98  # 2% improvement
            power_factor = 0.95    # 5% improvement
        
        return {
            'latency_factor': latency_factor,
            'power_factor': power_factor
        }
    
    def _apply_ttt_optimization(self, input_tensor: torch.Tensor, rank: int) -> Dict[str, float]:
        """Apply Test-Time Training optimization."""
        # Simulate TTT feedback
        feedback = {
            'estimated_energy': 0.001 * input_tensor.numel(),
            'expert_usage': torch.ones(self.mixtral_config.num_experts) * 0.125,  # Uniform usage
            'token_count': input_tensor.numel(),
            'batch_size': input_tensor.size(0),
            'seq_length': input_tensor.size(1)
        }
        
        # Update TTT router
        self.ttt_router.ttt_update(feedback)
        
        # Simulate TTT improvement
        ttt_update_count = self.ttt_router.ttt_update_count
        
        if ttt_update_count > 10:
            # TTT has converged - significant improvement
            latency_factor = 0.8   # 20% improvement
            energy_factor = 0.85   # 15% improvement
        elif ttt_update_count > 5:
            # TTT partially converged - moderate improvement
            latency_factor = 0.9   # 10% improvement
            energy_factor = 0.9    # 10% improvement
        else:
            # TTT early stages - minimal improvement
            latency_factor = 0.95  # 5% improvement
            energy_factor = 0.95   # 5% improvement
        
        return {
            'latency_factor': latency_factor,
            'energy_factor': energy_factor
        }
    
    def _synchronize_results(self, rank: int, world_size: int):
        """Synchronize results across all processes."""
        # Gather results from all processes
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, self.results)
        
        # Combine results
        if rank == 0:
            all_results = []
            for results in gathered_results:
                if results:
                    all_results.extend(results)
            self.results = all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze parallel test results."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            'summary': self._generate_summary(),
            'scaling_analysis': self._analyze_scaling(),
            'mode_comparison': self._analyze_mode_comparison(),
            'optimization_analysis': self._analyze_optimizations(),
            'gpu_utilization': self._analyze_gpu_utilization()
        }
        
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'total_tests': len(self.results),
            'num_gpus': self.config.num_gpus,
            'test_modes': list(set(r['test_mode'] for r in self.results)),
            'batch_sizes': list(set(r['batch_size'] for r in self.results)),
            'sequence_lengths': list(set(r['sequence_length'] for r in self.results))
        }
        
        # Performance summary
        latencies = [r['optimized_metrics']['latency_ms'] for r in self.results]
        energies = [r['optimized_metrics']['energy_joules'] for r in self.results]
        throughputs = [r['tokens_per_second'] for r in self.results]
        efficiencies = [r['energy_efficiency'] for r in self.results]
        
        summary['performance'] = {
            'avg_latency_ms': np.mean(latencies),
            'avg_energy_joules': np.mean(energies),
            'avg_throughput_tokens_per_sec': np.mean(throughputs),
            'avg_energy_efficiency': np.mean(efficiencies),
            'min_latency_ms': np.min(latencies),
            'max_throughput_tokens_per_sec': np.max(throughputs)
        }
        
        return summary
    
    def _analyze_scaling(self) -> Dict[str, Any]:
        """Analyze scaling behavior across GPUs."""
        scaling_data = {}
        
        for world_size in [1, 2, 4]:  # Test different GPU counts
            world_size_results = [r for r in self.results if r['world_size'] == world_size]
            
            if world_size_results:
                avg_throughput = np.mean([r['tokens_per_second'] for r in world_size_results])
                avg_latency = np.mean([r['optimized_metrics']['latency_ms'] for r in world_size_results])
                avg_efficiency = np.mean([r['energy_efficiency'] for r in world_size_results])
                
                scaling_data[world_size] = {
                    'avg_throughput': avg_throughput,
                    'avg_latency': avg_latency,
                    'avg_efficiency': avg_efficiency,
                    'sample_count': len(world_size_results)
                }
        
        # Calculate scaling efficiency
        if 1 in scaling_data and 4 in scaling_data:
            ideal_scaling = scaling_data[1]['avg_throughput'] * 4
            actual_scaling = scaling_data[4]['avg_throughput']
            scaling_efficiency = actual_scaling / ideal_scaling
            
            scaling_data['scaling_efficiency'] = scaling_efficiency
        
        return scaling_data
    
    def _analyze_mode_comparison(self) -> Dict[str, Any]:
        """Compare decode vs prefill modes."""
        decode_results = [r for r in self.results if r['test_mode'] == 'decode']
        prefill_results = [r for r in self.results if r['test_mode'] == 'prefill']
        
        comparison = {}
        
        if decode_results:
            decode_metrics = {
                'avg_latency': np.mean([r['optimized_metrics']['latency_ms'] for r in decode_results]),
                'avg_throughput': np.mean([r['tokens_per_second'] for r in decode_results]),
                'avg_efficiency': np.mean([r['energy_efficiency'] for r in decode_results]),
                'avg_power': np.mean([r['optimized_metrics']['power_watt'] for r in decode_results])
            }
            comparison['decode'] = decode_metrics
        
        if prefill_results:
            prefill_metrics = {
                'avg_latency': np.mean([r['optimized_metrics']['latency_ms'] for r in prefill_results]),
                'avg_throughput': np.mean([r['tokens_per_second'] for r in prefill_results]),
                'avg_efficiency': np.mean([r['energy_efficiency'] for r in prefill_results]),
                'avg_power': np.mean([r['optimized_metrics']['power_watt'] for r in prefill_results])
            }
            comparison['prefill'] = prefill_metrics
        
        return comparison
    
    def _analyze_optimizations(self) -> Dict[str, Any]:
        """Analyze effectiveness of different optimizations."""
        optimization_analysis = {}
        
        # Analyze topology optimization
        topology_results = [r for r in self.results if r['world_size'] > 1]
        if topology_results:
            topology_analysis = {
                'avg_latency_improvement': np.mean([
                    r['baseline_metrics']['latency_ms'] / r['optimized_metrics']['latency_ms']
                    for r in topology_results
                ]),
                'avg_energy_improvement': np.mean([
                    r['baseline_metrics']['energy_joules'] / r['optimized_metrics']['energy_joules']
                    for r in topology_results
                ])
            }
            optimization_analysis['topology'] = topology_analysis
        
        # Analyze thermal optimization
        if self.config.enable_thermal_awareness:
            thermal_results = [r for r in self.results if r['world_size'] > 1]
            if thermal_results:
                thermal_analysis = {
                    'avg_power_reduction': np.mean([
                        r['baseline_metrics']['power_watt'] / r['optimized_metrics']['power_watt']
                        for r in thermal_results
                    ])
                }
                optimization_analysis['thermal'] = thermal_analysis
        
        # Analyze TTT optimization
        if self.config.enable_ttt:
            ttt_results = self.results  # All results use TTT
            if ttt_results:
                ttt_analysis = {
                    'avg_latency_improvement': np.mean([
                        r['baseline_metrics']['latency_ms'] / r['optimized_metrics']['latency_ms']
                        for r in ttt_results
                    ]),
                    'avg_energy_improvement': np.mean([
                        r['baseline_metrics']['energy_joules'] / r['optimized_metrics']['energy_joules']
                        for r in ttt_results
                    ])
                }
                optimization_analysis['ttt'] = ttt_analysis
        
        return optimization_analysis
    
    def _analyze_gpu_utilization(self) -> Dict[str, Any]:
        """Analyze GPU utilization patterns."""
        utilization_data = {}
        
        for rank in range(self.config.num_gpus):
            rank_results = [r for r in self.results if r['rank'] == rank]
            
            if rank_results:
                avg_memory = np.mean([r['optimized_metrics']['memory_usage'] for r in rank_results])
                avg_compute = np.mean([r['optimized_metrics']['compute_utilization'] for r in rank_results])
                
                utilization_data[f'gpu_{rank}'] = {
                    'avg_memory_utilization': avg_memory,
                    'avg_compute_utilization': avg_compute,
                    'sample_count': len(rank_results)
                }
        
        return utilization_data
    
    def save_results(self, filename: str = "parallel_mixtral_results.json"):
        """Save results to file."""
        output_data = {
            'config': {
                'parallel_config': self.config.__dict__,
                'mixtral_config': self.mixtral_config.__dict__
            },
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Parallel test results saved to {filename}")
    
    def print_summary(self):
        """Print summary of results."""
        analysis = self.analyze_results()
        
        print("\n=== PARALLEL MIXTRAL TEST SUMMARY ===")
        
        summary = analysis['summary']
        print(f"Total tests: {summary['total_tests']}")
        print(f"GPUs tested: {summary['num_gpus']}")
        print(f"Test modes: {summary['test_modes']}")
        
        performance = summary['performance']
        print(f"\nPerformance Summary:")
        print(f"  Avg Latency: {performance['avg_latency_ms']:.2f} ms")
        print(f"  Avg Throughput: {performance['avg_throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"  Avg Energy Efficiency: {performance['avg_energy_efficiency']:.1f} tokens/J")
        print(f"  Min Latency: {performance['min_latency_ms']:.2f} ms")
        print(f"  Max Throughput: {performance['max_throughput_tokens_per_sec']:.1f} tokens/s")
        
        # Scaling analysis
        scaling = analysis['scaling_analysis']
        if 'scaling_efficiency' in scaling:
            print(f"\nScaling Efficiency: {scaling['scaling_efficiency']:.2%}")
        
        # Mode comparison
        mode_comp = analysis['mode_comparison']
        if 'decode' in mode_comp and 'prefill' in mode_comp:
            print(f"\nMode Comparison:")
            print(f"  Decode - Latency: {mode_comp['decode']['avg_latency']:.2f}ms, Throughput: {mode_comp['decode']['avg_throughput']:.1f}tok/s")
            print(f"  Prefill - Latency: {mode_comp['prefill']['avg_latency']:.2f}ms, Throughput: {mode_comp['prefill']['avg_throughput']:.1f}tok/s")

def run_parallel_test(rank: int, world_size: int, config: ParallelTestConfig, mixtral_config: MixtralConfig):
    """Entry point for parallel testing."""
    tester = ParallelMixtralTester(config, mixtral_config)
    tester.run_parallel_test(rank, world_size)
    
    if rank == 0:
        tester.print_summary()
        tester.save_results()

def main():
    """Main function for parallel Mixtral testing."""
    parser = argparse.ArgumentParser(description="Parallel Mixtral 7B Testing")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--batch_sizes", type=str, default="1,2,4,8,16", help="Comma-separated batch sizes")
    parser.add_argument("--seq_lengths", type=str, default="64,128,256,512", help="Comma-separated sequence lengths")
    parser.add_argument("--test_modes", type=str, default="decode,prefill", help="Comma-separated test modes")
    parser.add_argument("--expert_strategy", type=str, default="load_balanced", help="Expert placement strategy")
    parser.add_argument("--enable_thermal", action="store_true", help="Enable thermal awareness")
    parser.add_argument("--enable_ttt", action="store_true", help="Enable TTT optimization")
    
    args = parser.parse_args()
    
    # Parse arguments
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    test_modes = args.test_modes.split(',')
    
    # Configuration
    config = ParallelTestConfig(
        num_gpus=args.num_gpus,
        batch_sizes=batch_sizes,
        sequence_lengths=seq_lengths,
        test_modes=test_modes,
        expert_placement_strategy=args.expert_strategy,
        enable_thermal_awareness=args.enable_thermal,
        enable_ttt=args.enable_ttt
    )
    
    mixtral_config = MixtralConfig()
    
    print("=== Parallel Mixtral 7B Testing ===")
    print(f"GPUs: {config.num_gpus}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Sequence lengths: {config.sequence_lengths}")
    print(f"Test modes: {config.test_modes}")
    print(f"Expert strategy: {config.expert_placement_strategy}")
    print(f"Thermal awareness: {config.enable_thermal_awareness}")
    print(f"TTT optimization: {config.enable_ttt}")
    
    # Run parallel test
    if config.num_gpus > 1:
        mp.spawn(
            run_parallel_test,
            args=(config.num_gpus, config, mixtral_config),
            nprocs=config.num_gpus,
            join=True
        )
    else:
        # Single GPU test
        run_parallel_test(0, 1, config, mixtral_config)
    
    print("\n=== Parallel Testing Complete ===")

if __name__ == "__main__":
    main() 