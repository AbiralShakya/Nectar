# src/experiments/test_experts.py
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import math

# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.moe_models import MoEConfig, SwiGLUExpert, OptimizedQuantizedExpert, LaCTMoEExpert
from routers import AdaptiveRouter, RoutingStrategy, HardwareMetrics # Not directly used but good to have
from src.kernelcostmodel import KernelCostModel # Not directly used but good to have
from src.monitor import GpuSystemMonitor # Not directly used but good to have


print("--- Testing Expert Modules ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = MoEConfig(d_model=128, num_experts=2, top_k=1, expert_type="swiglu_ffn", batch_size=32).to(device) 
# Note: MoEConfig should NOT have .to(device)
config = MoEConfig(d_model=128, num_experts=2, top_k=1, expert_type="swiglu_ffn", batch_size=32)

dummy_input = torch.randn(config.batch_size, config.d_model, device=device) # Assuming CUDA

# --- Test SwiGLUExpert ---
print("\n--- Testing SwiGLUExpert ---")
expert1 = SwiGLUExpert(config, 0).cuda()
out1 = expert1(dummy_input)
print(f"  SwiGLUExpert output shape: {out1.shape}")
assert out1.shape == dummy_input.shape
print("SwiGLUExpert passed.")

# --- Test OptimizedQuantizedExpert ---
print("\n--- Testing OptimizedQuantizedExpert ---")
expert2 = OptimizedQuantizedExpert(config, 1).cuda() # Removed quantization_bits=8
out2 = expert2(dummy_input)
print(f"  OptimizedQuantizedExpert output shape: {out2.shape}")
assert out2.shape == dummy_input.shape
assert out2.dtype == dummy_input.dtype # Expecting it to return original dtype
print("OptimizedQuantizedExpert passed.")

# --- Test LaCTMoEExpert (TTT expert) ---
print("\n--- Testing LaCTMoEExpert ---")
lact_expert = LaCTMoEExpert(config, 0).cuda()

# Initial norm of fast weights
initial_lact_w1_norm = lact_expert.fast_weight_net.w1.norm().item()
initial_lact_w2_norm = lact_expert.fast_weight_net.w2.norm().item()
initial_lact_w3_norm = lact_expert.fast_weight_net.w3.norm().item()
print(f"  Initial LaCT w1 norm: {initial_lact_w1_norm:.6f}")
print(f"  Initial LaCT w2 norm: {initial_lact_w2_norm:.6f}")
print(f"  Initial LaCT w3 norm: {initial_lact_w3_norm:.6f}")

# Simulate a forward pass to trigger a fast weight update
# Make sure enough tokens are passed to trigger an update based on lact_chunk_size
num_tokens_to_simulate = config.lact_chunk_size * 2 # Ensure at least two chunks for buffer accumulation
# dummy_input_for_lact = torch.randn(num_tokens_to_simulate, config.d_model, device=device)
# Make it a dummy input that will cause a change.
dummy_input_for_lact = torch.ones(num_tokens_to_simulate, config.d_model, device=device) # Using ones for predictable gradients
# Add some variety to ensure gradients are not zero due to symmetry.
dummy_input_for_lact[:, 0] = torch.arange(num_tokens_to_simulate, dtype=torch.float32, device=device) / num_tokens_to_simulate

# Call forward multiple times to fill buffer and trigger update
for i in range(num_tokens_to_simulate // config.batch_size):
    start_idx = i * config.batch_size
    end_idx = start_idx + config.batch_size
    batch_x = dummy_input_for_lact[start_idx:end_idx]
    
    _ = lact_expert(batch_x) # Pass a batch of tokens

# Explicitly trigger an update if it wasn't already (for testing robustness)
# This part is for debugging: ensure the update function is callable.
print("\n--- Manually triggering LaCT update for verification ---")
# Before manually triggering, ensure buffers are populated
if not lact_expert.chunk_buffer_k:
    print("  Chunk buffer is empty, simulating another pass to fill it.")
    _ = lact_expert(dummy_input_for_lact[0:config.batch_size]) # Send one more batch

# Temporarily enable requires_grad for weights to check their gradients directly
# This is for debugging purposes, to see if gradients are calculated.
lact_expert.fast_weight_net.w1.requires_grad_(True)
lact_expert.fast_weight_net.w2.requires_grad_(True)
lact_expert.fast_weight_net.w3.requires_grad_(True)

# Get current norms before manual step
norm_before_manual_step_w1 = lact_expert.fast_weight_net.w1.norm().item()
norm_before_manual_step_w2 = lact_expert.fast_weight_net.w2.norm().item()
norm_before_manual_step_w3 = lact_expert.fast_weight_net.w3.norm().item()

if lact_expert.chunk_buffer_k:
    print("  Performing internal LaCT update...")
    # This will call _perform_lact_update internally if conditions are met
    # or we can call it directly for debugging:
    # lact_expert._perform_lact_update()
    # It's better to let forward() handle it for a realistic test flow.
    # The above `_ = lact_expert(batch_x)` loop should have handled it.

    # After the loop, let's just assert that a change happened.
    # If the loop above didn't trigger it, then the logic in `forward` needs fixing for chunking.
    # Let's ensure `tokens_since_last_update` and `current_chunk_tokens_count` are properly reset/managed.

    # Re-check the norms AFTER the forward passes that should have triggered updates
    updated_lact_w1_norm = lact_expert.fast_weight_net.w1.norm().item()
    updated_lact_w2_norm = lact_expert.fast_weight_net.w2.norm().item()
    updated_lact_w3_norm = lact_expert.fast_weight_net.w3.norm().item()

    print(f"  Updated LaCT w1 norm: {updated_lact_w1_norm:.6f}")
    print(f"  Updated LaCT w2 norm: {updated_lact_w2_norm:.6f}")
    print(f"  Updated LaCT w3 norm: {updated_lact_w3_norm:.6f}")

    # The actual assertion
    # Use a small epsilon for floating point comparison
    epsilon = 1e-6
    assert abs(updated_lact_w1_norm - initial_lact_w1_norm) > epsilon, "LaCT w1 did not change significantly."
    assert abs(updated_lact_w2_norm - initial_lact_w2_norm) > epsilon, "LaCT w2 did not change significantly."
    assert abs(updated_lact_w3_norm - initial_lact_w3_norm) > epsilon, "LaCT w3 did not change significantly."

    print("LaCTMoEExpert fast weights updated successfully.")

    # Reset requires_grad_ to False for subsequent operations if any
    lact_expert.fast_weight_net.w1.requires_grad_(False)
    lact_expert.fast_weight_net.w2.requires_grad_(False)
    lact_expert.fast_weight_net.w3.requires_grad_(False)

else:
    print("  LaCT update was NOT triggered because chunk buffer was empty. Check chunking logic.")

print("--- All Expert Modules Tests Complete ---")