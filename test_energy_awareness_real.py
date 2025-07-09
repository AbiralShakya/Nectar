#!/usr/bin/env python3
"""
Test that actually demonstrates energy-aware routing.
Creates synthetic stress by making some experts artificially expensive.
"""

import torch
import numpy as np
from models.ttt_router import EnergyAwareTTTRouter

def test_energy_awareness():
    """Test if the router actually avoids expensive experts."""
    print("=== Energy-Aware Routing Test ===")
    
    # Parameters
    d_model = 768
    num_experts = 16
    top_k = 2
    batch_size = 8
    seq_length = 64
    
    print(f"Testing with {num_experts} experts, top_k={top_k}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    print()
    
    # Test different lambda values
    lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    for lambda_energy in lambda_values:
        print(f"--- Testing lambda_energy = {lambda_energy} ---")
        
        # Create router
        router = EnergyAwareTTTRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            lambda_energy=lambda_energy
        )
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, seq_length, d_model)
        
        # Track expert usage over multiple batches
        expert_usage_total = torch.zeros(num_experts)
        energy_consumed = 0.0
        
        # Run multiple batches to see adaptation
        for batch_idx in range(50):
            # Create synthetic energy feedback where some experts are expensive
            # Make experts 0-3 expensive (high energy), others cheap (low energy)
            expensive_experts = [0, 1, 2, 3]
            cheap_experts = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            
            # Simulate energy consumption based on expert usage
            expert_usage = torch.zeros(num_experts)
            
            # Run router
            expert_indices, expert_weights, router_metadata = router(input_tensor)
            
            # Calculate actual expert usage
            for i in range(num_experts):
                expert_usage[i] = (expert_indices == i).sum().item()
            
            # Calculate energy based on which experts were used
            batch_energy = 0.0
            for i in range(num_experts):
                if expert_usage[i] > 0:
                    if i in expensive_experts:
                        batch_energy += expert_usage[i] * 10.0  # Expensive experts
                    else:
                        batch_energy += expert_usage[i] * 1.0   # Cheap experts
            
            energy_consumed += batch_energy
            
            # TTT update with energy feedback
            feedback = {
                'estimated_energy': batch_energy,
                'expert_usage': expert_usage,
                'token_count': input_tensor.numel(),
                'batch_size': batch_size,
                'seq_length': seq_length
            }
            router.ttt_update(feedback)
            
            # Accumulate usage
            expert_usage_total += expert_usage
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                expensive_usage = sum(expert_usage_total[i] for i in expensive_experts)
                cheap_usage = sum(expert_usage_total[i] for i in cheap_experts)
                total_usage = expert_usage_total.sum()
                
                print(f"  Batch {batch_idx}: "
                      f"Expensive={expensive_usage:.0f} ({expensive_usage/total_usage*100:.1f}%), "
                      f"Cheap={cheap_usage:.0f} ({cheap_usage/total_usage*100:.1f}%), "
                      f"Energy={batch_energy:.1f}J")
        
        # Final analysis
        total_usage = expert_usage_total.sum()
        expensive_usage = sum(expert_usage_total[i] for i in expensive_experts)
        cheap_usage = sum(expert_usage_total[i] for i in cheap_experts)
        
        expensive_ratio = expensive_usage / total_usage
        cheap_ratio = cheap_usage / total_usage
        
        print(f"  Final Results:")
        print(f"    Expensive experts (0-3): {expensive_usage:.0f} tokens ({expensive_ratio*100:.1f}%)")
        print(f"    Cheap experts (4-15): {cheap_usage:.0f} tokens ({cheap_ratio*100:.1f}%)")
        print(f"    Total energy: {energy_consumed:.1f}J")
        
        # Expected: expensive_ratio should decrease with higher lambda
        expected_expensive_ratio = 4.0 / 16.0  # 25% if random
        energy_awareness_score = (expected_expensive_ratio - expensive_ratio) / expected_expensive_ratio * 100
        
        if energy_awareness_score > 10:
            print(f"    ✓ Energy-aware! Avoided {energy_awareness_score:.1f}% of expensive experts")
        elif energy_awareness_score > 0:
            print(f"    ⚠ Slightly energy-aware ({energy_awareness_score:.1f}% avoidance)")
        else:
            print(f"    ❌ Not energy-aware (using {abs(energy_awareness_score):.1f}% more expensive experts)")
        
        print()

if __name__ == "__main__":
    test_energy_awareness() 