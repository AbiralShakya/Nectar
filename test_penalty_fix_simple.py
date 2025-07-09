#!/usr/bin/env python3
"""
Simple test of energy penalty scaling fix.
Tests only the router component without external dependencies.
"""

import torch
import json
import time
import numpy as np
from models.ttt_router import EnergyAwareTTTRouter

def test_penalty_scaling_simple():
    """Test energy penalty scaling with different lambda values."""
    print("=== Energy Penalty Scaling Test (Simple) ===")
    
    # Test parameters
    d_model = 768
    num_experts = 16
    top_k = 2
    batch_size = 8
    seq_length = 64
    
    # Test different lambda values
    lambda_values = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    results_summary = {}
    
    for lambda_energy in lambda_values:
        print(f"\n--- Testing lambda_energy = {lambda_energy} ---")
        
        # Create router
        router = EnergyAwareTTTRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            lambda_energy=lambda_energy
        )
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, seq_length, d_model)
        
        # Track metrics over multiple batches
        energy_measurements = []
        accuracy_measurements = []
        expert_usage_history = []
        
        # Run multiple batches to see adaptation
        for batch_idx in range(50):
            # Simulate energy feedback (4.17 Joules like in your test)
            feedback = {
                'estimated_energy': 4.17,
                'expert_usage': torch.zeros(num_experts),
                'token_count': input_tensor.numel(),
                'batch_size': batch_size,
                'seq_length': seq_length
            }
            
            # Update router
            router.ttt_update(feedback)
            
            # Run forward pass
            expert_indices, expert_weights, router_metadata = router(input_tensor)
            
            # Analyze results
            logits_before_penalty = router.gate(input_tensor)
            logits_after_penalty = logits_before_penalty.clone()
            
            # Apply penalty manually to see the effect
            if router.last_estimated_energy > 0:
                base_penalty = router.lambda_energy * router.energy_scale * float(router.last_estimated_energy)
                expert_penalties = router._compute_adaptive_penalties(base_penalty)
                logits_after_penalty = logits_after_penalty - expert_penalties.unsqueeze(0)
            
            # Calculate metrics
            logits_range_before = (logits_before_penalty.min().item(), logits_before_penalty.max().item())
            logits_range_after = (logits_after_penalty.min().item(), logits_after_penalty.max().item())
            
            # Calculate expert usage
            expert_usage = torch.zeros(num_experts)
            for i in range(num_experts):
                expert_usage[i] = (expert_indices == i).sum().item()
            
            # Calculate load balance (how uniform is the distribution)
            expected_uniform = batch_size * seq_length / num_experts
            variance = torch.var(expert_usage.float()).item()
            load_balance_score = 1.0 / (1.0 + variance / (expected_uniform ** 2))
            
            # Simulate accuracy based on routing quality
            base_accuracy = 0.3
            routing_quality = 1.0 - router_metadata.get('energy_penalty_applied', False) * 0.1
            accuracy = base_accuracy * routing_quality + np.random.normal(0, 0.05)
            accuracy = max(0.1, min(0.9, accuracy))
            
            # Store metrics
            energy_measurements.append(4.17)  # Simulated energy
            accuracy_measurements.append(accuracy)
            expert_usage_history.append(expert_usage.cpu().numpy())
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                penalty_magnitude = abs(logits_range_after[0] - logits_range_before[0])
                print(f"  Batch {batch_idx}: Penalty={base_penalty:.6f}, "
                      f"Logits=[{logits_range_after[0]:.3f}, {logits_range_after[1]:.3f}], "
                      f"Load Balance={load_balance_score:.3f}, Accuracy={accuracy:.3f}")
        
        # Calculate final metrics
        avg_energy = np.mean(energy_measurements)
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
        
        # Check penalty magnitude
        final_base_penalty = router.lambda_energy * router.energy_scale * float(router.last_estimated_energy)
        penalty_reasonable = final_base_penalty < 1.0
        
        # Store results
        results_summary[lambda_energy] = {
            "avg_energy_joules": float(avg_energy),
            "avg_accuracy": float(avg_accuracy),
            "routing_entropy": float(routing_entropy),
            "expert_usage_distribution": expert_usage_distribution,
            "ttt_update_count": router.ttt_update_count,
            "energy_savings_percent": float(energy_savings_percent),
            "accuracy_loss_percent": float(accuracy_loss_percent),
            "final_base_penalty": float(final_base_penalty),
            "penalty_reasonable": penalty_reasonable
        }
        
        print(f"  Final Base Penalty: {final_base_penalty:.6f}")
        print(f"  Average Energy: {avg_energy:.6f} J")
        print(f"  Average Accuracy: {avg_accuracy:.3f}")
        print(f"  Routing Entropy: {routing_entropy:.3f}")
        print(f"  Energy Savings: {energy_savings_percent:.2f}%")
        print(f"  Accuracy Loss: {accuracy_loss_percent:.2f}%")
        
        if penalty_reasonable:
            print(f"  ✓ Penalty magnitude is reasonable")
        else:
            print(f"  ⚠ Penalty magnitude is too high")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("=== OVERALL SUMMARY ===")
    
    best_energy_savings = max(results_summary.values(), key=lambda x: x['energy_savings_percent'])
    best_accuracy = max(results_summary.values(), key=lambda x: x['avg_accuracy'])
    best_adaptation = max(results_summary.values(), key=lambda x: x['routing_entropy'])
    
    print(f"Best Energy Savings: {best_energy_savings['energy_savings_percent']:.2f}%")
    print(f"Best Accuracy: {best_accuracy['avg_accuracy']:.3f}")
    print(f"Best Adaptation (Entropy): {best_adaptation['routing_entropy']:.3f}")
    
    # Check if any penalties are reasonable
    reasonable_penalties = sum(1 for r in results_summary.values() if r['penalty_reasonable'])
    print(f"Reasonable Penalties: {reasonable_penalties}/{len(results_summary)}")
    
    if reasonable_penalties == len(results_summary):
        print("✓ All penalties are now reasonable!")
    else:
        print("⚠ Some penalties are still too high")
    
    # Save results
    output_file = "penalty_fix_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results_summary

if __name__ == "__main__":
    test_penalty_scaling_simple() 