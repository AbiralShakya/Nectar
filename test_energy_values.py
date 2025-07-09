#!/usr/bin/env python3
"""
Test to see what energy values the kernel cost model is returning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.kernelcostmodel import KernelCostModel
    print("✓ KernelCostModel imported successfully")
except ImportError as e:
    print(f"✗ Error importing KernelCostModel: {e}")
    sys.exit(1)

def test_energy_values():
    """Test what energy values the kernel cost model returns."""
    print("=== Testing Kernel Cost Model Energy Values ===")
    
    # Initialize cost model
    cost_model = KernelCostModel(gpu_type="A100")
    
    # Test different operations and batch sizes
    test_cases = [
        ("moe_router", 8),
        ("ffn_gate", 8),
        ("ffn_gate", 512),  # Total tokens for batch_size=8, seq_length=64
        ("attention_qk", 8),
        ("layer_norm", 8),
    ]
    
    print("\nEnergy values from kernel cost model:")
    print("-" * 60)
    
    total_energy = 0.0
    
    for op_type, batch_size in test_cases:
        try:
            cost = cost_model.get_cost(op_type, batch_size)
            energy_joules = cost["energy_joules"]
            latency_ms = cost["latency_ms"]
            
            print(f"{op_type:15} (batch_size={batch_size:3d}): "
                  f"Energy={energy_joules:8.6f}J, "
                  f"Latency={latency_ms:6.3f}ms")
            
            total_energy += energy_joules
            
        except Exception as e:
            print(f"{op_type:15} (batch_size={batch_size:3d}): ERROR - {e}")
    
    print("-" * 60)
    print(f"Total energy for all operations: {total_energy:.6f}J")
    
    # Calculate what the penalty would be with our current scaling
    lambda_energy = 0.05
    energy_scale = 0.01  # Our current fix
    
    base_penalty = lambda_energy * energy_scale * total_energy
    print(f"\nPenalty calculation:")
    print(f"  lambda_energy = {lambda_energy}")
    print(f"  energy_scale = {energy_scale}")
    print(f"  total_energy = {total_energy:.6f}J")
    print(f"  base_penalty = {base_penalty:.6f}")
    
    if base_penalty < 1.0:
        print("✓ Penalty is reasonable")
    else:
        print("⚠ Penalty is still too high!")
    
    # Test what the penalty would be with the old scaling
    old_energy_scale = 1000.0
    old_base_penalty = lambda_energy * old_energy_scale * total_energy
    print(f"  old_energy_scale = {old_energy_scale}")
    print(f"  old_base_penalty = {old_base_penalty:.6f}")
    
    # Suggest better scaling
    suggested_scale = 0.001  # Much smaller
    suggested_penalty = lambda_energy * suggested_scale * total_energy
    print(f"\nSuggested fix:")
    print(f"  suggested_scale = {suggested_scale}")
    print(f"  suggested_penalty = {suggested_penalty:.6f}")
    
    if suggested_penalty < 0.1:
        print("✓ Suggested penalty is very reasonable")
    else:
        print("⚠ Suggested penalty might still be too high")

if __name__ == "__main__":
    test_energy_values() 