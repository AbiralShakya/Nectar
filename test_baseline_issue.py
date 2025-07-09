#!/usr/bin/env python3
"""
Test to demonstrate the baseline energy calculation issue.
"""

import math

def demonstrate_baseline_issue():
    """Show why the baseline energy calculation is wrong."""
    print("=== Baseline Energy Calculation Issue ===")
    
    # Parameters
    batch_size = 8
    seq_length = 64
    total_tokens = batch_size * seq_length  # 512 tokens
    num_experts = 16
    top_k = 2
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Total tokens: {total_tokens}")
    print(f"Number of experts: {num_experts}")
    print(f"Top-k: {top_k}")
    print()
    
    # Simulate energy costs (from kernel cost model)
    routing_energy_per_batch = 0.001  # Small routing cost
    expert_energy_per_token = 0.004   # Larger expert cost
    
    # Current baseline calculation (WRONG)
    baseline_routing_energy = routing_energy_per_batch
    baseline_expert_energy = total_tokens * expert_energy_per_token  # All tokens to experts
    baseline_total = baseline_routing_energy + baseline_expert_energy
    
    print("=== Current (Wrong) Baseline ===")
    print(f"Baseline routing energy: {baseline_routing_energy:.6f}J")
    print(f"Baseline expert energy: {baseline_expert_energy:.6f}J")
    print(f"Baseline total: {baseline_total:.6f}J")
    print()
    
    # Actual energy calculation (CORRECT)
    actual_routing_energy = routing_energy_per_batch
    actual_expert_tokens = total_tokens * top_k / num_experts  # Only routed tokens
    actual_expert_energy = actual_expert_tokens * expert_energy_per_token
    actual_total = actual_routing_energy + actual_expert_energy
    
    print("=== Actual Energy (Correct) ===")
    print(f"Actual routing energy: {actual_routing_energy:.6f}J")
    print(f"Actual expert tokens: {actual_expert_tokens:.1f}")
    print(f"Actual expert energy: {actual_expert_energy:.6f}J")
    print(f"Actual total: {actual_total:.6f}J")
    print()
    
    # Calculate "energy savings" with wrong baseline
    wrong_energy_savings = ((baseline_total - actual_total) / baseline_total) * 100
    
    print("=== Energy Savings Calculation ===")
    print(f"Wrong baseline: {baseline_total:.6f}J")
    print(f"Actual energy: {actual_total:.6f}J")
    print(f"Energy savings: {wrong_energy_savings:.2f}%")
    print()
    
    if wrong_energy_savings < 0:
        print("🚨 PROBLEM: Energy savings is negative because baseline is wrong!")
        print("   The baseline assumes ALL tokens go to experts, but only some do.")
        print("   This makes the system look like it's using MORE energy when it's not.")
    else:
        print("✓ Energy savings is positive (but this is misleading)")
    
    # Correct baseline calculation
    correct_baseline = actual_total  # Baseline should be the same as actual
    correct_energy_savings = ((correct_baseline - actual_total) / correct_baseline) * 100
    
    print(f"\n=== Correct Calculation ===")
    print(f"Correct baseline: {correct_baseline:.6f}J")
    print(f"Actual energy: {actual_total:.6f}J")
    print(f"Energy savings: {correct_energy_savings:.2f}%")
    print()
    
    print("=== Recommendation ===")
    print("The baseline should be calculated the same way as the actual energy:")
    print("1. Use the same routing logic (top-k selection)")
    print("2. Only count tokens that are actually routed to experts")
    print("3. Then compare different lambda_energy values against each other")

if __name__ == "__main__":
    demonstrate_baseline_issue() 