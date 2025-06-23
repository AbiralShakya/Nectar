import torch
from src.moe_models import MoEConfig, OptimizedMoELayer, SwiGLUExpert, OptimizedQuantizedExpert, LaCTMoEExpert
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor 

print("Testing OptimizedMoELayer...")
kcm = KernelCostModel(data_path="kernel_cost_models/kernel_cost_model_d4096.json", gpu_type="A100") # Ensure exists
monitor = GpuSystemMonitor(device_id=0) # Will simulate if no pynvml

# Test with SwiGLU experts
config_swiglu = MoEConfig(d_model=4096, num_experts=4, top_k=1, expert_type="swiglu_ffn").cuda()
moe_layer_swiglu = OptimizedMoELayer(config_swiglu, kcm, monitor).cuda()
dummy_input = torch.randn(16, config_swiglu.d_model).cuda()

output_swiglu, metrics_swiglu = moe_layer_swiglu(dummy_input, use_adaptive_routing=False) # Use standard router
print(f"SwiGLU MoE Layer output shape: {output_swiglu.shape}")
print(f"  Energy Loss: {metrics_swiglu['aux_losses']['energy_loss'].item():.4f}")
print(f"  Expert Timings: {metrics_swiglu['expert_timings']}")

# Test with Quantized experts
config_quant = MoEConfig(d_model=4096, num_experts=4, top_k=1, expert_type="quantized", quantization_bits=8).cuda()
moe_layer_quant = OptimizedMoELayer(config_quant, kcm, monitor).cuda()
output_quant, metrics_quant = moe_layer_quant(dummy_input, use_adaptive_routing=False)
print(f"Quantized MoE Layer output shape: {output_quant.shape}")
print(f"  Energy Loss (Quantized): {metrics_quant['aux_losses']['energy_loss'].item():.4f}")
assert output_quant.dtype == dummy_input.dtype # Ensure type consistency

# Test with LaCT experts
config_lact = MoEConfig(d_model=4096, num_experts=4, top_k=1, expert_type="lact_expert", lact_chunk_size=16, lact_lr=1e-3).cuda()
moe_layer_lact = OptimizedMoELayer(config_lact, kcm, monitor).cuda()

# For LaCT expert, need to ensure internal updates are triggered
# Run multiple times to exceed chunk size
print("\nRunning LaCTMoE Expert through OptimizedMoELayer (multiple batches):")
for i in range(5): # This will trigger internal LaCT updates
    batch_input = torch.randn(16, config_lact.d_model).cuda()
    output_lact, metrics_lact = moe_layer_lact(batch_input, use_adaptive_routing=False)
    print(f"  Batch {i+1} LaCT MoE Layer output shape: {output_lact.shape}")
    if metrics_lact['expert_timings']: # Check if any experts ran
         lact_expert_id = list(metrics_lact['expert_timings'].keys())[0] # Just take first
         if isinstance(moe_layer_lact.experts[lact_expert_id], LaCTMoEExpert):
             if moe_layer_lact.experts[lact_expert_id].current_chunk_tokens_count == 0 and i > 0:
                 print(f"    LaCT Expert (ID {lact_expert_id}) internal update triggered in batch {i+1}.")

monitor.stop() # Clean up monitor threads