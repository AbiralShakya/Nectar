#!/usr/bin/env python3
"""
Simple test to estimate energy values and penalty scaling.
"""

import math

def estimate_energy_values():
    """Estimate energy values based on the kernel cost model logic."""
    print("=== Energy Value Estimation ===")
    
    # GPU specs for A100
    peak_power_w = 400
    compute_throughput_tflops = 312
    memory_bandwidth_gb_s = 1935
    
    # Operation specs (from kernel cost model)
    operations = {
        "moe_router": {"flops_per_token": 2048, "memory_intensity": 1.0},
        "ffn_gate": {"flops_per_token": 33554432, "memory_intensity": 0.8},
        "attention_qk": {"flops_per_token": 4096, "memory_intensity": 2.0},
        "layer_norm": {"flops_per_token": 8192, "memory_intensity": 2.5},
    }
    
    batch_size = 8
    seq_length = 64
    total_tokens = batch_size * seq_length  # 512 tokens
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Total tokens: {total_tokens}")
    print()
    
    total_energy = 0.0
    
    for op_name, op_specs in operations.items():
        flops_per_token = op_specs["flops_per_token"]
        memory_intensity = op_specs["memory_intensity"]
        
        # Calculate energy for this operation
        if op_name == "ffn_gate":
            # For expert computation, use total tokens
            tokens = total_tokens
        else:
            # For routing, use batch size
            tokens = batch_size
        
        total_flops = flops_per_token * tokens
        memory_bytes = flops_per_token * memory_intensity * 2 * tokens  # BF16 = 2 bytes
        
        # Energy calculation (simplified)
        compute_time_s = total_flops / (compute_throughput_tflops * 1e12)
        memory_time_s = memory_bytes / (memory_bandwidth_gb_s * 1e9)
        
        latency_s = max(compute_time_s, memory_time_s)
        
        # Energy = Power × Time
        compute_energy = compute_time_s * (peak_power_w * 0.6)  # 60% for compute
        memory_energy = memory_time_s * (peak_power_w * 0.3)   # 30% for memory
        static_energy = latency_s * (peak_power_w * 0.1)       # 10% static
        
        total_op_energy = compute_energy + memory_energy + static_energy
        
        print(f"{op_name:15} ({tokens:3d} tokens): "
              f"Energy={total_op_energy:8.6f}J, "
              f"Latency={latency_s*1000:6.3f}ms")
        
        total_energy += total_op_energy
    
    print("-" * 60)
    print(f"Total energy: {total_energy:.6f}J")
    
    # Test different penalty scales
    lambda_energy = 0.05
    
    scales_to_test = [
        ("Current fix", 0.01),
        ("Old (broken)", 1000.0),
        ("Suggested 1", 0.001),
        ("Suggested 2", 0.0001),
        ("Very small", 0.00001),
    ]
    
    print(f"\nPenalty calculations (lambda_energy = {lambda_energy}):")
    print("-" * 60)
    
    for name, scale in scales_to_test:
        penalty = lambda_energy * scale * total_energy
        print(f"{name:15}: scale={scale:8.5f}, penalty={penalty:8.6f}")
        
        if penalty < 0.1:
            print("  ✓ Very reasonable")
        elif penalty < 1.0:
            print("  ✓ Reasonable")
        elif penalty < 10.0:
            print("  ⚠ High")
        else:
            print("  🚨 Too high!")
    
    # Suggest optimal scale
    print(f"\nRecommendation:")
    optimal_scale = 0.0001  # This gives penalty ~0.02
    optimal_penalty = lambda_energy * optimal_scale * total_energy
    print(f"Use energy_scale = {optimal_scale}")
    print(f"This gives penalty = {optimal_penalty:.6f} (very reasonable)")

if __name__ == "__main__":
    estimate_energy_values() 