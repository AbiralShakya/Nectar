#!/usr/bin/env python3
"""
Simple test to verify energy penalty scaling fix.
This script tests the router with different lambda values to ensure
the penalty doesn't overwhelm the logits.
"""

import torch
import numpy as np
from models.ttt_router import EnergyAwareTTTRouter

def test_penalty_scaling():
    """Test energy penalty scaling with different lambda values."""
    print("=== Energy Penalty Scaling Test ===")
    
    # Test parameters
    d_model = 768
    num_experts = 16
    top_k = 2
    batch_size = 8
    seq_length = 64
    
    # Test different lambda values
    lambda_values = [0.001, 0.01, 0.05, 0.1, 0.2]
    
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
        
        print(f"  Base penalty: {base_penalty:.6f}")
        print(f"  Logits before penalty: [{logits_range_before[0]:.3f}, {logits_range_before[1]:.3f}]")
        print(f"  Logits after penalty: [{logits_range_after[0]:.3f}, {logits_range_after[1]:.3f}]")
        print(f"  Load balance score: {load_balance_score:.3f}")
        print(f"  Expert usage variance: {variance:.3f}")
        
        # Check if penalty is reasonable
        penalty_magnitude = abs(logits_range_after[0] - logits_range_before[0])
        if penalty_magnitude < 5.0:
            print(f"  ✓ Penalty magnitude is reasonable ({penalty_magnitude:.3f})")
        else:
            print(f"  ⚠ Penalty magnitude is too high ({penalty_magnitude:.3f})")
        
        # Check if routing is still functional
        if logits_range_after[1] - logits_range_after[0] > 1.0:
            print(f"  ✓ Routing logits have sufficient range")
        else:
            print(f"  ⚠ Routing logits are too compressed")

if __name__ == "__main__":
    test_penalty_scaling() 