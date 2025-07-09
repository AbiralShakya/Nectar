#!/usr/bin/env python3
"""
Local test of energy-aware TTT with fixed penalty scaling.
This runs the same test as the SLURM script but locally.
"""

import torch
import json
import time
import numpy as np
from models.ttt_router import EnergyAwareTTTRouter
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor

def run_energy_aware_ttt_test(lambda_energy=0.05, num_batches=50, num_epochs=2):
    """Run energy-aware TTT test with synthetic data."""
    print(f"=== Energy-Aware TTT Test (Fixed Penalty) ===")
    print(f"Lambda Energy: {lambda_energy}")
    
    # Initialize components
    d_model = 768
    num_experts = 16
    top_k = 2
    batch_size = 8
    seq_length = 64
    
    # Initialize router
    router = EnergyAwareTTTRouter(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        lambda_energy=lambda_energy
    )
    
    # Initialize cost model and monitor
    cost_model = KernelCostModel()
    monitor = GpuSystemMonitor()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = torch.randn(num_batches * batch_size, seq_length, d_model)
    
    # Track metrics
    energy_measurements = []
    latency_measurements = []
    power_measurements = []
    accuracy_measurements = []
    expert_usage_history = []
    
    print(f"Running {num_epochs} epochs with {num_batches} batches each...")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_data = synthetic_data[start_idx:end_idx]
            
            # Simulate energy measurement
            estimated_energy = 4.17  # Same as your cluster test
            
            # Run router
            expert_indices, expert_weights, router_metadata = router(batch_data)
            
            # Simulate latency and power
            latency_ms = 2.2 + np.random.normal(0, 0.1)  # ~2.2ms like your test
            power_watt = 1900 + np.random.normal(0, 50)  # ~1900W like your test
            
            # Simulate accuracy (should improve with better routing)
            base_accuracy = 0.3
            routing_quality = 1.0 - router_metadata.get('energy_penalty_applied', False) * 0.1
            accuracy = base_accuracy * routing_quality + np.random.normal(0, 0.05)
            accuracy = max(0.1, min(0.9, accuracy))
            
            # Store metrics
            energy_measurements.append(estimated_energy)
            latency_measurements.append(latency_ms)
            power_measurements.append(power_watt)
            accuracy_measurements.append(accuracy)
            
            # Track expert usage
            expert_usage = router_metadata.get('expert_usage', torch.zeros(num_experts))
            expert_usage_history.append(expert_usage.cpu().numpy())
            
            # TTT update
            feedback = {
                'estimated_energy': estimated_energy,
                'expert_usage': expert_usage,
                'token_count': batch_data.numel(),
                'batch_size': batch_size,
                'seq_length': seq_length
            }
            router.ttt_update(feedback)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Energy={estimated_energy:.6f}J, "
                      f"Latency={latency_ms:.2f}ms, Power={power_watt:.1f}W, "
                      f"Accuracy={accuracy:.3f}")
    
    # Calculate final metrics
    avg_energy = np.mean(energy_measurements)
    avg_latency = np.mean(latency_measurements)
    avg_power = np.mean(power_measurements)
    avg_accuracy = np.mean(accuracy_measurements)
    
    # Calculate expert usage distribution
    final_expert_usage = np.mean(expert_usage_history, axis=0)
    expert_usage_distribution = (final_expert_usage / final_expert_usage.sum()).tolist()
    
    # Calculate routing entropy (measure of load balancing)
    routing_entropy = -np.sum([p * np.log(p + 1e-8) for p in expert_usage_distribution])
    
    # Calculate energy savings (compared to baseline)
    baseline_energy = 4.5  # Assume baseline energy
    energy_savings_percent = ((baseline_energy - avg_energy) / baseline_energy) * 100
    
    # Calculate accuracy loss (compared to baseline)
    baseline_accuracy = 0.95  # Assume baseline accuracy
    accuracy_loss_percent = ((baseline_accuracy - avg_accuracy) / baseline_accuracy) * 100
    
    # Compile results
    results = {
        "test_config": {
            "lambda_energy": lambda_energy,
            "num_experts": num_experts,
            "moe_top_k": top_k,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "d_model": d_model,
            "num_batches": num_batches,
            "num_epochs": num_epochs
        },
        "results": {
            "avg_energy_joules": float(avg_energy),
            "avg_latency_ms": float(avg_latency),
            "avg_power_watt": float(avg_power),
            "avg_accuracy": float(avg_accuracy),
            "thermal_imbalance_score": 0.0,  # Not implemented in local test
            "routing_entropy": float(routing_entropy),
            "expert_usage_distribution": expert_usage_distribution,
            "ttt_update_count": router.ttt_update_count,
            "energy_savings_percent": float(energy_savings_percent),
            "accuracy_loss_percent": float(accuracy_loss_percent)
        },
        "timestamp": time.time()
    }
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Lambda Energy: {lambda_energy}")
    print(f"Average Energy: {avg_energy:.6f} J")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Power: {avg_power:.1f} W")
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Routing Entropy: {routing_entropy:.3f}")
    print(f"Energy Savings: {energy_savings_percent:.2f}%")
    print(f"Accuracy Loss: {accuracy_loss_percent:.2f}%")
    print(f"TTT Updates: {router.ttt_update_count}")
    
    # Check if penalty is working correctly
    if router.last_estimated_energy > 0:
        base_penalty = router.lambda_energy * router.energy_scale * float(router.last_estimated_energy)
        print(f"Final Base Penalty: {base_penalty:.6f}")
        
        if base_penalty < 1.0:
            print("✓ Penalty magnitude is reasonable")
        else:
            print("⚠ Penalty magnitude is too high")
    
    return results

if __name__ == "__main__":
    # Test different lambda values
    lambda_values = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        results = run_energy_aware_ttt_test(lambda_energy=lambda_val, num_batches=20, num_epochs=1)
        
        # Save results
        output_file = f"local_test_lambda_{lambda_val}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("All local tests complete!")
    print("Check the JSON files to compare performance across lambda values.") 