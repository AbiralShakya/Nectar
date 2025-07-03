#!/usr/bin/env python3
"""
Quick test to verify energy penalty is working correctly.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ttt_router import EnergyAwareTTTRouter

def test_energy_penalty():
    print("Testing Energy-Aware TTT Router Energy Penalty...")
    
    # Create router
    d_model = 768
    num_experts = 8
    top_k = 2
    
    # Test different lambda values
    for lambda_energy in [0.0, 0.01, 0.1, 1.0]:
        print(f"\n=== Testing lambda_energy = {lambda_energy} ===")
        
        router = EnergyAwareTTTRouter(d_model, num_experts, top_k, lambda_energy)
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, d_model)
        
        # Test without energy penalty
        indices1, probs1, meta1 = router(x)
        print(f"Without energy penalty:")
        print(f"  Expert usage: {meta1['expert_usage'].tolist()}")
        print(f"  Diversity: {(meta1['expert_usage'] > 0).float().mean().item():.3f}")
        
        # Apply energy feedback
        feedback = {
            'estimated_energy': 0.001,  # 1 mJ
            'expert_usage': torch.ones(num_experts)  # All experts used equally
        }
        router.ttt_update(feedback)
        
        # Test with energy penalty
        indices2, probs2, meta2 = router(x)
        print(f"With energy penalty:")
        print(f"  Expert usage: {meta2['expert_usage'].tolist()}")
        print(f"  Diversity: {(meta2['expert_usage'] > 0).float().mean().item():.3f}")
        print(f"  Energy penalty applied: {meta2['energy_penalty_applied']}")
        
        # Check if penalty affected routing
        usage_diff = (meta2['expert_usage'] - meta1['expert_usage']).abs().sum()
        print(f"  Usage change: {usage_diff.item()}")
        
        if lambda_energy > 0 and usage_diff > 0:
            print("  ✓ Energy penalty affected routing!")
        elif lambda_energy == 0:
            print("  ✓ No penalty applied (as expected)")
        else:
            print("  ⚠ Energy penalty had no effect")

if __name__ == "__main__":
    test_energy_penalty() 