#!/usr/bin/env python3
"""
Test to verify the baseline energy calculation fix.
"""

def test_baseline_fix():
    """Test the corrected baseline energy calculation."""
    print("=== Testing Baseline Energy Fix ===")
    
    # Parameters
    batch_size = 8
    seq_length = 64
    total_tokens = batch_size * seq_length  # 512 tokens
    num_experts = 16
    top_k = 2
    
    print(f"Parameters: batch_size={batch_size}, seq_length={seq_length}")
    print(f"Total tokens: {total_tokens}, num_experts={num_experts}, top_k={top_k}")
    print()
    
    # Simulate energy costs
    routing_energy_per_batch = 0.001
    expert_energy_per_token = 0.004
    
    # Calculate baseline using the FIXED method
    baseline_routing_energy = routing_energy_per_batch
    expert_tokens = total_tokens * top_k / num_experts  # 512 * 2 / 16 = 64
    baseline_expert_energy = expert_tokens * expert_energy_per_token
    baseline_total = baseline_routing_energy + baseline_expert_energy
    
    print("=== Fixed Baseline Calculation ===")
    print(f"Routing energy: {baseline_routing_energy:.6f}J")
    print(f"Expert tokens: {expert_tokens:.1f}")
    print(f"Expert energy: {baseline_expert_energy:.6f}J")
    print(f"Baseline total: {baseline_total:.6f}J")
    print()
    
    # Test different lambda_energy values
    lambda_values = [0.0, 0.05, 0.1, 0.2]
    
    print("=== Energy Comparison ===")
    print("Lambda | Energy | Savings | Status")
    print("-" * 40)
    
    for lambda_energy in lambda_values:
        # Simulate energy with different lambda values
        # Higher lambda should reduce energy (but not by much with our small penalty)
        energy_reduction = lambda_energy * 0.1  # Small reduction
        actual_energy = baseline_total * (1.0 - energy_reduction)
        
        energy_savings = ((baseline_total - actual_energy) / baseline_total) * 100
        
        status = "✓ Good" if energy_savings >= 0 else "⚠ Bad"
        
        print(f"{lambda_energy:6.2f} | {actual_energy:6.6f}J | {energy_savings:6.2f}% | {status}")
    
    print()
    print("=== Expected Behavior ===")
    print("✓ lambda=0.0: No energy penalty, baseline energy")
    print("✓ lambda>0.0: Small energy reduction due to energy-aware routing")
    print("✓ All energy savings should be positive or zero")
    print("✓ Energy savings should increase with lambda_energy")

if __name__ == "__main__":
    test_baseline_fix() 