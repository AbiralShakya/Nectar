#!/usr/bin/env python3
"""
Dynamic Expert Rerouting Experiment

This script tests the new DYNAMIC_EXPERT_REROUTING strategy that:
1. Tracks historical batch distribution patterns
2. Predicts future imbalances based on trends
3. Reroutes tokens to distribute load more evenly
4. Optimizes for power/energy efficiency rather than just performance

Based on the requirements:
- Uses previous batch distribution and uses that in future distribution
- Implements expert dynamic rerouting for MoE imbalance
- Optimizes for joules per token (energy efficiency)
- Considers thermal consumption and power budgets
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import time
import numpy as np
from typing import Dict, Tuple, Any, List
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.moe_models import MoEConfig, MoETransformerBlock
from routers import RoutingStrategy, AdaptiveRouter, BatchDistributionTracker
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from data_utils import DataLoaderManager
from metrics_logger import MetricsLogger

def create_imbalanced_workload(dataloader_manager: DataLoaderManager, 
                              imbalance_factor: float = 0.8) -> torch.utils.data.DataLoader:
    """
    Create a workload that naturally leads to imbalanced expert usage.
    This simulates real-world scenarios where certain experts are over-utilized.
    """
    # Get standard workload
    standard_loader = dataloader_manager.get_workload("standard", batch_size=32, num_samples=1000)
    
    # For now, we'll use the standard loader but the imbalance will come from
    # the routing decisions and the batch distribution tracker's analysis
    return standard_loader

def run_dynamic_expert_rerouting_experiment(args):
    """
    Main experiment function for testing dynamic expert rerouting.
    """
    print("=== Dynamic Expert Rerouting Experiment ===")
    print(f"Testing strategy: {RoutingStrategy.DYNAMIC_EXPERT_REROUTING.value}")
    
    # Setup device
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    moe_config = MoEConfig(
        d_model=args.d_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        expert_type=args.expert_type,
        batch_size=args.batch_size
    )
    
    kernel_cost_model = KernelCostModel(data_path=args.kernel_cost_path)
    gpu_monitor = GpuSystemMonitor(device_id=args.device_id)
    
    # Create model with dynamic expert rerouting
    model = MoETransformerBlock(
        moe_config, kernel_cost_model, gpu_monitor
    ).to(device)
    
    # Set the routing strategy to dynamic expert rerouting
    model.moe_layer.adaptive_router.strategy = RoutingStrategy.DYNAMIC_EXPERT_REROUTING
    
    # Initialize data loader
    dataloader_manager = DataLoaderManager(
        data_dir=args.data_dir,
        d_model=args.d_model,
        seq_length=args.seq_length
    )
    
    # Create metrics logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dynamic_expert_rerouting_{timestamp}.csv"
    metrics_logger = MetricsLogger(log_filename)
    
    # Create imbalanced workload
    dataloader = create_imbalanced_workload(dataloader_manager, imbalance_factor=0.8)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    
    print(f"\nStarting experiment with {args.epochs} epochs...")
    print(f"Model config: {moe_config.num_experts} experts, {moe_config.top_k} top-k")
    print(f"Expert type: {moe_config.expert_type}")
    
    # Track distribution statistics over time
    distribution_stats = []
    rerouting_effectiveness = []
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_distributions = []
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass with dynamic expert rerouting
            batch_start_time = time.perf_counter()
            output, metrics = model(x, use_adaptive_routing=True)
            batch_end_time = time.perf_counter()
            
            # Calculate performance metrics
            inference_latency_ms = (batch_end_time - batch_start_time) * 1000.0
            throughput_tokens_per_sec = x.size(0) * moe_config.d_model / (batch_end_time - batch_start_time)
            
            # Get routing information
            routing_info = metrics.get("routing_metadata", {})
            expert_usage = metrics.get("expert_usage", torch.zeros(moe_config.num_experts))
            
            # Calculate task loss
            task_loss = criterion(output, y)
            
            # Get auxiliary losses
            aux_losses_dict = metrics.get("aux_losses", {})
            energy_loss = aux_losses_dict.get("energy_loss", torch.tensor(0.0, device=device))
            load_balance_loss = aux_losses_dict.get("load_balance_loss", torch.tensor(0.0, device=device))
            router_z_loss = aux_losses_dict.get("router_z_loss", torch.tensor(0.0, device=device))
            
            # Total loss with energy awareness
            total_loss = task_loss + \
                         moe_config.load_balance_weight * load_balance_loss + \
                         moe_config.router_z_loss_weight * router_z_loss + \
                         0.001 * energy_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Get current GPU stats
            gpu_stats = gpu_monitor.get_current_stats()
            
            # Get batch distribution tracker statistics
            bdt_stats = model.moe_layer.adaptive_router.batch_distribution_tracker.get_statistics()
            
            # Log comprehensive metrics
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "epoch": epoch,
                "batch": batch_idx,
                "strategy": RoutingStrategy.DYNAMIC_EXPERT_REROUTING.value,
                "expert_type": moe_config.expert_type,
                "loss": total_loss.item(),
                "task_loss": task_loss.item(),
                "energy_loss": energy_loss.item(),
                "load_balance_loss": load_balance_loss.item(),
                "router_z_loss": router_z_loss.item(),
                "inference_latency_ms": inference_latency_ms,
                "throughput_tokens_per_sec": throughput_tokens_per_sec,
                "gpu_temperature_c": gpu_stats['temperature'],
                "gpu_power_watt": gpu_stats['power_watt'],
                "gpu_thermal_state": gpu_stats['thermal_state'],
                "gpu_utilization_percent": gpu_stats['gpu_utilization_percent'],
                "memory_utilization_percent": gpu_stats['memory_utilization_percent'],
                "expert_usage_counts": expert_usage.tolist(),
                "routing_entropy": metrics.get("routing_entropy", 0.0),
                
                # Batch distribution tracker metrics
                "avg_imbalance_score": bdt_stats.get('avg_imbalance_score', 0.0),
                "max_imbalance_score": bdt_stats.get('max_imbalance_score', 0.0),
                "min_imbalance_score": bdt_stats.get('min_imbalance_score', 0.0),
                "rerouting_count": bdt_stats.get('rerouting_count', 0),
                "avg_rerouting_improvement": bdt_stats.get('avg_rerouting_improvement', 0.0),
                "history_length": bdt_stats.get('history_length', 0),
                "expert_usage_trends": bdt_stats.get('expert_usage_trends', []),
                
                # Rerouting metadata if available
                "current_imbalance": routing_info.get('current_imbalance', 0.0),
                "predicted_imbalance": routing_info.get('predicted_imbalance', 0.0),
                "confidence": routing_info.get('confidence', 0.0),
                "needs_rerouting": routing_info.get('needs_rerouting', False),
                "thermal_pressure": routing_info.get('thermal_pressure', 0.0),
                "power_pressure": routing_info.get('power_pressure', 0.0),
                "rerouting_strength": routing_info.get('rerouting_strength', 0.0),
            }
            
            metrics_logger.log(log_data)
            
            # Store distribution statistics
            epoch_distributions.append({
                'batch': batch_idx,
                'expert_usage': expert_usage.cpu().numpy(),
                'imbalance_score': bdt_stats.get('avg_imbalance_score', 0.0),
                'rerouting_active': routing_info.get('needs_rerouting', False)
            })
            
            epoch_losses.append(total_loss.item())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: "
                      f"Loss: {total_loss.item():.4f}, "
                      f"Imbalance: {bdt_stats.get('avg_imbalance_score', 0):.3f}, "
                      f"Temp: {gpu_stats['temperature']:.1f}°C, "
                      f"Power: {gpu_stats['power_watt']:.1f}W, "
                      f"Rerouting: {routing_info.get('needs_rerouting', False)}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_loss = np.mean(epoch_losses)
        
        # Calculate distribution improvement
        if len(epoch_distributions) > 1:
            early_imbalance = epoch_distributions[0]['imbalance_score']
            late_imbalance = epoch_distributions[-1]['imbalance_score']
            improvement = early_imbalance - late_imbalance
            rerouting_effectiveness.append(improvement)
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. "
              f"Avg loss: {avg_loss:.4f}, "
              f"Distribution improvement: {improvement if 'improvement' in locals() else 0:.3f}")
        
        # Store epoch statistics
        distribution_stats.append({
            'epoch': epoch,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'avg_imbalance': np.mean([d['imbalance_score'] for d in epoch_distributions]),
            'rerouting_count': sum(1 for d in epoch_distributions if d['rerouting_active'])
        })
    
    # Final analysis
    print("\n=== Experiment Results ===")
    
    # Analyze distribution improvements
    if rerouting_effectiveness:
        avg_improvement = np.mean(rerouting_effectiveness)
        print(f"Average distribution improvement per epoch: {avg_improvement:.3f}")
    
    # Analyze final statistics
    final_bdt_stats = model.moe_layer.adaptive_router.batch_distribution_tracker.get_statistics()
    print(f"Final imbalance score: {final_bdt_stats.get('avg_imbalance_score', 0):.3f}")
    print(f"Total rerouting count: {final_bdt_stats.get('rerouting_count', 0)}")
    print(f"Average rerouting improvement: {final_bdt_stats.get('avg_rerouting_improvement', 0):.3f}")
    
    # Save detailed results
    results = {
        'experiment_config': {
            'strategy': RoutingStrategy.DYNAMIC_EXPERT_REROUTING.value,
            'num_experts': moe_config.num_experts,
            'top_k': moe_config.top_k,
            'expert_type': moe_config.expert_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        'final_statistics': final_bdt_stats,
        'distribution_stats': distribution_stats,
        'rerouting_effectiveness': rerouting_effectiveness
    }
    
    results_filename = f"dynamic_expert_rerouting_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_filename}")
    print(f"Metrics logged to: {log_filename}")
    
    # Cleanup
    gpu_monitor.stop()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Dynamic Expert Rerouting Experiment")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--expert_type", type=str, default="simple", 
                       choices=["simple", "quantized"], help="Expert type")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    
    # Hardware configuration
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--kernel_cost_path", type=str, 
                       default="energy/cost_table.json", help="Kernel cost model path")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_dynamic_expert_rerouting_experiment(args)
    
    print("\n=== Experiment Summary ===")
    print("Dynamic expert rerouting experiment completed successfully!")
    print("The system should have:")
    print("1. Tracked historical batch distribution patterns")
    print("2. Predicted future imbalances based on trends")
    print("3. Applied rerouting biases to balance expert usage")
    print("4. Optimized for power/energy efficiency")
    print("\nCheck the generated CSV and JSON files for detailed analysis.")

if __name__ == "__main__":
    main() 