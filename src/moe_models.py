# src/moe_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time # For expert timing in SimpleMoELayer
from typing import Dict, Tuple, Any
from routers import AdaptiveRouter
# Placeholder for KernelCostModel and AdaptiveRouter from other modules
# These imports assume the files are in the same 'src' directory or accessible via Python path
# Corrected: KernelCostModel is defined in moe_models.py itself for now as per previous iteration.
# GpuSystemMonitor and AdaptiveRouter need to be imported/forward declared.
from monitor import GpuSystemMonitor
from kernelcostmodel import KernelCostModel
from routers import KernelCostModel, GpuSystemMonitor, AdaptiveRouter, RoutingStrategy
# AdaptiveRouter is a forward declaration in SimpleMoELayer.
# --- Expert Definitions ---

class SimpleExpert(nn.Module):
    """
    A basic Feed-Forward Network acting as an MoE expert.
    Used for initial kernel profiling of standard operations.
    """
    def __init__(self, d_model: int, expert_id: int):
        super().__init__()
        self.expert_id = expert_id # Useful for debugging/profiling
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # These are the operations whose kernels we will profile
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class QuantizedExpert(nn.Module):
    # ... (existing __init__ and parameters) ...

    def _pack_weights(self):
        """
        Simulates quantization and packing of full-precision weights into uint8 buffers.
        This method handles `self.quantization_bits` (2 or 4).
        It maps signed integers to unsigned for storage, and applies MSB toggling concept for 4-bit.
        """
        # Ensure full precision weights are on CPU for packing, then move packed to GPU buffer later
        fc1_weight_full_cpu = self.fc1_weight_full.data.cpu()
        fc2_weight_full_cpu = self.fc2_weight_full.data.cpu()

        # Define min/max values for the target bit-width (signed)
        min_val_signed = -(2**(self.quantization_bits - 1)) # e.g., -8 for 4-bit, -2 for 2-bit
        max_val_signed = (2**(self.quantization_bits - 1)) - 1 # e.g., 7 for 4-bit, 1 for 2-bit

        # --- FC1 Packing ---
        # 1. Calculate scales (per-row, float16)
        max_val_per_row_fc1, _ = fc1_weight_full_cpu.abs().max(dim=1, keepdim=True)
        scales_fc1 = max_val_per_row_fc1 / (max_val_signed + 1e-5) # Add epsilon to avoid div by zero
        self.fc1_scales.copy_(scales_fc1.to(torch.float16)) # Store FP16 scales

        # 2. Quantize and clamp to signed integer range
        quant_weights_fc1 = (fc1_weight_full_cpu / (self.fc1_scales.cpu() + 1e-5)).round().to(torch.int8)
        quant_weights_fc1 = torch.clamp(quant_weights_fc1, min_val_signed, max_val_signed)

        # 3. Pack into uint8 buffer based on bit-width
        packed_buffer_fc1 = torch.zeros(self.fc1_weight_packed.numel(), dtype=torch.uint8)

        if self.quantization_bits == 4:
            # Each byte holds two 4-bit values. Example: byte = (high_nibble << 4) | low_nibble
            # We map signed_val (-8 to 7) to unsigned_val (0 to 15) for storage: unsigned_val = signed_val + 8
            # The order in the paper (w0, w16, w1, w17...) is for SIMD-aware packing, 
            # here we do a simpler consecutive packing for simulation.
            
            # Reshape to (rows, cols/2, 2)
            num_elements_per_byte = 8 // self.quantization_bits # 2 for 4-bit
            quant_weights_reshaped = quant_weights_fc1.view(self.fc1_hidden_dim, -1, num_elements_per_byte)
            
            # Apply MSB toggling concept for efficient signed unpacking later (optional, but good for realism)
            # If MSB is 0 (positive range), add 0x8. If MSB is 1 (negative range), subtract 0x8
            # Or simpler: map -8 to 7 to 0 to 15 directly
            # For simplicity, let's map -8 to 7 -> 0 to 15 for storage (add 8)
            val0_unsigned = (quant_weights_reshaped[:, :, 0] + 8).to(torch.uint8)
            val1_unsigned = (quant_weights_reshaped[:, :, 1] + 8).to(torch.uint8)
            
            packed_fc1_data = (val0_unsigned << 4) | (val1_unsigned & 0x0F)
            packed_buffer_fc1.copy_(packed_fc1_data.flatten())

        elif self.quantization_bits == 2:
            # Each byte holds four 2-bit values. Example: byte = (v3 << 6) | (v2 << 4) | (v1 << 2) | v0
            # Map signed_val (-2 to 1) to unsigned_val (0 to 3): unsigned_val = signed_val + 2
            
            num_elements_per_byte = 8 // self.quantization_bits # 4 for 2-bit
            quant_weights_reshaped = quant_weights_fc1.view(self.fc1_hidden_dim, -1, num_elements_per_byte)

            val0_unsigned = (quant_weights_reshaped[:, :, 0] + 2).to(torch.uint8)
            val1_unsigned = (quant_weights_reshaped[:, :, 1] + 2).to(torch.uint8)
            val2_unsigned = (quant_weights_reshaped[:, :, 2] + 2).to(torch.uint8)
            val3_unsigned = (quant_weights_reshaped[:, :, 3] + 2).to(torch.uint8)

            packed_fc1_data = (val0_unsigned << 6) | (val1_unsigned << 4) | (val2_unsigned << 2) | (val3_unsigned & 0x03)
            packed_buffer_fc1.copy_(packed_fc1_data.flatten())

        self.fc1_weight_packed.copy_(packed_buffer_fc1.to(self.fc1_weight_packed.device)) # Move to device

        # --- FC2 Packing (similar logic) ---
        weight_fp16 = self.fc2_weight_full.data.cpu().to(torch.float16)

        max_val_per_row_fc2, _ = weight_fp16.abs().max(dim=1, keepdim=True)
        scales_fc2 = max_val_per_row_fc2 / (max_val_signed + 1e-5)
        self.fc2_scales.copy_(scales_fc2.to(torch.float16))

        quant_weights_fc2 = (weight_fp16 / (self.fc2_scales.cpu() + 1e-5)).round().to(torch.int8)
        quant_weights_fc2 = torch.clamp(quant_weights_fc2, min_val_signed, max_val_signed)

        packed_buffer_fc2 = torch.zeros(self.fc2_weight_packed.numel(), dtype=torch.uint8)

        if self.quantization_bits == 4:
            num_elements_per_byte = 8 // self.quantization_bits
            quant_weights_reshaped = quant_weights_fc2.view(self.fc2_output_dim, -1, num_elements_per_byte)
            val0_unsigned = (quant_weights_reshaped[:, :, 0] + 8).to(torch.uint8)
            val1_unsigned = (quant_weights_reshaped[:, :, 1] + 8).to(torch.uint8)
            packed_fc2_data = (val0_unsigned << 4) | (val1_unsigned & 0x0F)
            packed_buffer_fc2.copy_(packed_fc2_data.flatten())
        elif self.quantization_bits == 2:
            num_elements_per_byte = 8 // self.quantization_bits
            quant_weights_reshaped = quant_weights_fc2.view(self.fc2_output_dim, -1, num_elements_per_byte)
            val0_unsigned = (quant_weights_reshaped[:, :, 0] + 2).to(torch.uint8)
            val1_unsigned = (quant_weights_reshaped[:, :, 1] + 2).to(torch.uint8)
            val2_unsigned = (quant_weights_reshaped[:, :, 2] + 2).to(torch.uint8)
            val3_unsigned = (quant_weights_reshaped[:, :, 3] + 2).to(torch.uint8)
            packed_fc2_data = (val0_unsigned << 6) | (val1_unsigned << 4) | (val2_unsigned << 2) | (val3_unsigned & 0x03)
            packed_buffer_fc2.copy_(packed_fc2_data.flatten())
        
        self.fc2_weight_packed.copy_(packed_buffer_fc2.to(self.fc2_weight_packed.device)) # Move to device

    def _dequantize_and_unpack(self, packed_weights: torch.Tensor, scales: torch.Tensor,
                               d_out: int, d_in: int, quantization_bits: int) -> torch.Tensor:
        """
        Simulates the dequantization and unpacking process on the GPU using PyTorch ops.
        Handles both 4-bit and 2-bit packing formats as defined in _pack_weights.
        """
        packed_weights = packed_weights.to(scales.device) # Ensure packed_weights are on GPU

        unpacked_tensor = torch.empty(d_out, d_in, device=scales.device, dtype=scales.dtype)

        if quantization_bits == 4:
            # Unpack two 4-bit values from each uint8 byte
            # These are currently 0-15. Subtract 8 to get -8 to 7.
            high_nibbles_unsigned = (packed_weights >> 4).to(torch.int8)
            low_nibbles_unsigned = (packed_weights & 0x0F).to(torch.int8)

            high_nibbles_signed = high_nibbles_unsigned - 8
            low_nibbles_signed = low_nibbles_unsigned - 8
            
            num_packed_elements_per_row = d_in // 2
            high_nibbles_reshaped = high_nibbles_signed.view(d_out, num_packed_elements_per_row)
            low_nibbles_reshaped = low_nibbles_signed.view(d_out, num_packed_elements_per_row)
            
            # Fill into the target unpacked_tensor, interleaving columns
            unpacked_tensor[:, 0::2] = high_nibbles_reshaped.float()
            unpacked_tensor[:, 1::2] = low_nibbles_reshaped.float()

        elif quantization_bits == 2:
            # Unpack four 2-bit values from each uint8 byte
            # These are currently 0-3. Subtract 2 to get -2 to 1.
            val3_unsigned = (packed_weights >> 6) & 0x03
            val2_unsigned = (packed_weights >> 4) & 0x03
            val1_unsigned = (packed_weights >> 2) & 0x03
            val0_unsigned = (packed_weights & 0x03)

            val3_signed = val3_unsigned.to(torch.int8) - 2
            val2_signed = val2_unsigned.to(torch.int8) - 2
            val1_signed = val1_unsigned.to(torch.int8) - 2
            val0_signed = val0_unsigned.to(torch.int8) - 2

            num_packed_elements_per_row = d_in // 4
            val3_reshaped = val3_signed.view(d_out, num_packed_elements_per_row)
            val2_reshaped = val2_signed.view(d_out, num_packed_elements_per_row)
            val1_reshaped = val1_signed.view(d_out, num_packed_elements_per_row)
            val0_reshaped = val0_signed.view(d_out, num_packed_elements_per_row)

            # Fill into the target unpacked_tensor, interleaving columns
            unpacked_tensor[:, 0::4] = val3_reshaped.float()
            unpacked_tensor[:, 1::4] = val2_reshaped.float()
            unpacked_tensor[:, 2::4] = val1_reshaped.float()
            unpacked_tensor[:, 3::4] = val0_reshaped.float()

        else:
            raise ValueError(f"Dequantization for {quantization_bits}-bit not implemented.")

        # Apply scales (scales are [d_out, 1], will broadcast)
        return unpacked_tensor * scales.to(unpacked_tensor.dtype)

# --- MoE Layer and Block Definitions ---
# These remain largely the same, but with correct imports and expert instantiation.

class SimpleMoELayer(nn.Module):
    def __init__(self, gate: nn.Module, experts: nn.ModuleList, top_k: int,
                 kernel_cost_model: KernelCostModel, # Added for Phase 2+
                 gpu_system_monitor: GpuSystemMonitor, # Added for Phase 2+
                 routing_strategy: str = "baseline"):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.n_experts = len(experts)
        self.top_k = top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        
        # We need to explicitly pass KernelCostModel and GpuSystemMonitor to the router
        # Router definition will be in routers.py, so an import is needed.
        self.router = AdaptiveRouter(self.n_experts, top_k,
                                     kernel_cost_model, gpu_system_monitor, # Pass the new dependencies
                                     routing_strategy)
        
        self.expert_cumulative_timings_ms: Dict[int, float] = {} # Total time spent by each expert
        self.metrics_buffer: Dict[str, Any] = {} # To store per-batch metrics for logging

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        num_tokens, d_model = x.shape
        device = x.device

        gate_logits = self.gate(x)
        
        # Pass num_tokens to router (needed for kernel_cost_model lookup based on batch_size)
        top_k_indices, top_k_probs, _ = self.router(gate_logits, num_tokens) # Router now returns 3 things
        gate_probs_all = F.softmax(gate_logits, dim=-1)

        # Aux loss for load balancing (standard MoE)
        top1_indices = top_k_indices[:, 0]
        expert_mask_top1 = F.one_hot(top1_indices, num_classes=self.n_experts).float()
        tokens_per_expert = expert_mask_top1.sum(dim=0)
        avg_gate_prob = gate_probs_all.mean(dim=0)
        aux_loss = (tokens_per_expert / (num_tokens + 1e-8) * avg_gate_prob).sum() * self.n_experts

        output = torch.zeros_like(x)
        expert_usage_counts = torch.zeros(self.n_experts, device=device)
        expert_batch_timings_ms: Dict[int, float] = {}

        # Expert dispatch and execution
        for expert_id in range(self.n_experts):
            expert_tokens_mask = (top_k_indices == expert_id).any(dim=-1)
            expert_token_indices = torch.where(expert_tokens_mask)[0]

            if expert_token_indices.numel() > 0:
                expert_input = x[expert_token_indices]

                expert_weights = torch.zeros(expert_token_indices.numel(), device=device, dtype=x.dtype)
                for i, token_idx in enumerate(expert_token_indices):
                    pos = torch.where(top_k_indices[token_idx] == expert_id)[0]
                    if pos.numel() > 0:
                        expert_weights[i] = top_k_probs[token_idx, pos].sum()

                start = time.perf_counter()
                expert_output = self.experts[expert_id](expert_input)
                duration_ms = (time.perf_counter() - start) * 1000.0
                expert_batch_timings_ms[expert_id] = duration_ms
                self.expert_cumulative_timings_ms[expert_id] += duration_ms

                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                output[expert_token_indices] += weighted_output
                expert_usage_counts[expert_id] = expert_token_indices.numel()

        # Collect metrics for current batch
        self.metrics_buffer = {
            "expert_usage_current": expert_usage_counts.cpu().numpy(),
            "total_assignments": expert_usage_counts.sum().item(),
            "expert_batch_timings_ms": expert_batch_timings_ms,
            "expert_cumulative_timings_ms": dict(self.expert_cumulative_timings_ms),
            "top_k_indices": top_k_indices.detach().cpu(),
            "top_k_probs": top_k_probs.detach().cpu()
        }

        return output, aux_loss, self.metrics_buffer


class MoETransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int,
                 kernel_cost_model: KernelCostModel, # Added
                 gpu_system_monitor: GpuSystemMonitor, # Added
                 routing_strategy: str = "baseline",
                 expert_type: str = "simple", # New arg to select expert type
                 quantization_bits: int = 4): # New arg for QuantizedExpert
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        self.routing_strategy = routing_strategy
        self.expert_type = expert_type
        self.quantization_bits = quantization_bits

        self.gate = nn.Linear(d_model, num_experts)

        if expert_type == "simple":
            experts = nn.ModuleList([SimpleExpert(d_model, i) for i in range(num_experts)])
        elif expert_type == "quantized":
            # Pass d_model as input_dim and output_dim, hidden_dim as d_model*2 for the expert
            experts = nn.ModuleList([QuantizedExpert(d_model, i, quantization_bits=quantization_bits) for i in range(num_experts)])
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

        self.moe_layer = SimpleMoELayer(self.gate, experts, top_k,
                                        kernel_cost_model, gpu_system_monitor,
                                        routing_strategy)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        return self.moe_layer(x)


# --- Helper for Energy Loss (can be in a separate utils file or here) ---
def compute_energy_loss(selected_expert_indices: torch.Tensor, top_k_probs: torch.Tensor,
                        kernel_cost_model: KernelCostModel, alpha: float = 0.001) -> torch.Tensor:
    """
    Computes a weighted energy loss based on activated experts and their routing probabilities,
    using the KernelCostModel for predicted costs.
    """
    # Ensure kernel_cost_model is available
    if kernel_cost_model is None:
        return torch.tensor(0.0, device=selected_expert_indices.device)

    total_predicted_energy = torch.tensor(0.0, device=selected_expert_indices.device)
    
    # Iterate through tokens and their selected experts/probabilities
    for i in range(selected_expert_indices.shape[0]): # For each token
        token_expert_indices = selected_expert_indices[i]
        token_expert_probs = top_k_probs[i]
        
        # When looking up kernel costs, use 1 for batch_size as each token is processed by an expert
        # effectively as a batch of 1. The inner operations then scale with d_model.
        effective_kernel_batch_size = 1 

        for j in range(token_expert_indices.shape[0]): # For each of the top_k experts selected for this token
            # expert_id = int(token_expert_indices[j].item()) # expert_id is not used in cost lookup here
            prob = token_expert_probs[j].item() # Probability of this token going to this expert

            # Sum the predicted costs of constituent kernels for ONE generic expert (assuming experts are homogeneous)
            # --- These op_type strings MUST match what you profile in expert_kernel_profiler.py ---
            
            expert_predicted_energy_cost_for_token = 0.0
            
            op_cost_dequant = kernel_cost_model.get_cost("dequant_unpack_op", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_dequant.get("energy_joules", 0.0)

            op_cost_fc1 = kernel_cost_model.get_cost("linear_fc1", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_fc1.get("energy_joules", 0.0)

            op_cost_relu = kernel_cost_model.get_cost("relu", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_relu.get("energy_joules", 0.0)
            
            op_cost_fc2 = kernel_cost_model.get_cost("linear_fc2", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_fc2.get("energy_joules", 0.0)
            
            # Accumulate weighted predicted energy
            total_predicted_energy += prob * expert_predicted_energy_cost_for_token

    return alpha * total_predicted_energy