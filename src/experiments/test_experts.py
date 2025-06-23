import torch
from src.moe_models import MoEConfig, SwiGLUExpert, OptimizedQuantizedExpert, LaCTMoEExpert, SwiGLUFastWeightNet

print("Testing Expert Modules...")
config = MoEConfig(d_model=128, num_experts=2, top_k=1, lact_chunk_size=16, lact_lr=1e-2, lact_fast_weight_dim_ratio=0.5)
dummy_input = torch.randn(config.batch_size, config.d_model).cuda() # Assuming CUDA

# Test SwiGLUExpert
expert1 = SwiGLUExpert(config, 0).cuda()
out1 = expert1(dummy_input)
print(f"SwiGLUExpert output shape: {out1.shape}, dtype: {out1.dtype}")
assert out1.shape == dummy_input.shape

# Test OptimizedQuantizedExpert
expert2 = OptimizedQuantizedExpert(config, 1, quantization_bits=8).cuda()
out2 = expert2(dummy_input)
print(f"OptimizedQuantizedExpert output shape: {out2.shape}, dtype: {out2.dtype}")
assert out2.shape == dummy_input.shape
assert out2.dtype == dummy_input.dtype # Expecting it to return original dtype

# Test SwiGLUFastWeightNet (isolated)
print("\nTesting SwiGLUFastWeightNet isolation...")
fw_net = SwiGLUFastWeightNet(config.d_model, int(config.d_model * config.lact_fast_weight_dim_ratio)).cuda()
initial_w1_norm = fw_net.w1.norm().item()

# Simulate update input
k_chunk = torch.randn(config.lact_chunk_size, config.d_model).cuda()
v_chunk = torch.randn(config.lact_chunk_size, config.d_model).cuda()
lr_coeffs_chunk = torch.ones(config.lact_chunk_size, 3).cuda()

grads = fw_net.compute_update_gradients(k_chunk, v_chunk, lr_coeffs_chunk)
print(f"  Gradients computed: w1_grad norm={grads['w1'].norm().item():.4f}")
# Need to simulate optimizer step for isolation test
optimizer = torch.optim.AdamW(fw_net.parameters(), lr=config.lact_lr)
optimizer.zero_grad() # Clear any previous grads

# Manually assign gradients
fw_net.w1.grad = grads['w1']
fw_net.w2.grad = grads['w2']
fw_net.w3.grad = grads['w3']

optimizer.step()

print(f"  w1 norm after isolated update: {fw_net.w1.norm().item():.4f} (should change)")
assert fw_net.w1.norm().item() != initial_w1_norm # Check if params changed

# Test LaCTMoEExpert
print("\nTesting LaCTMoEExpert...")
lact_expert = LaCTMoEExpert(config, 2).cuda()
initial_lact_w1_norm = lact_expert.fast_weight_net.w1.norm().item()
print(f"  Initial LaCT expert fast_weight_net.w1 norm: {initial_lact_w1_norm:.4f}")

# Simulate multiple forward passes to trigger chunk update
for i in range(5): # This will exceed chunk size (5 * 32 > 16)
    batch_input = torch.randn(config.batch_size, config.d_model).cuda()
    out_lact = lact_expert(batch_input)
    print(f"  Batch {i+1} LaCTExpert output shape: {out_lact.shape}")
    if lact_expert.current_chunk_tokens_count == 0 and i > 0:
        print(f"  LaCT expert update triggered after batch {i}. New fast_weight_net.w1 norm: {lact_expert.fast_weight_net.w1.norm().item():.4f}")
assert lact_expert.fast_weight_net.w1.norm().item() != initial_lact_w1_norm # Check if fast weights changed