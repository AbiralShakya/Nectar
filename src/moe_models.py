import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Tuple, Any
from routers import AdaptiveRouter
from thermal_signal import ThermalSignalGenerator


class KernelCostModel: 
    def __init__(self, data_path: str = None, data: dict = None):
        if data_path:
            import pandas as pd
            self.data = pd.read_json(data_path)
            self.data.set_index(["op_type", "batch_size"], inplace=True)
        elif data is not None:
            import pandas as pd
            self.data = pd.DataFrame(data)
            self.data.set_index(["op_type", "batch_size"], inplace=True)
        else:
            self.data = pd.DataFrame(columns=["op_type", "batch_size", "energy_joules", "latency_ms", "temp_impact"])
            self.data.set_index(["op_type", "batch_size"], inplace=True)

    def get_cost(self, op_type: str, batch_size: int) -> dict:
        try:
            return self.data.loc[(op_type, batch_size)].to_dict()
        except KeyError:
            # Fallback for missing data - crucial during early development
            # You should refine these defaults based on initial profiling if possible
            # These are simple dummy values for now
            energy = {'linear_fc1': 0.05, 'relu': 0.001, 'linear_fc2': 0.05}.get(op_type, 0.01) * (batch_size / 32)
            latency = {'linear_fc1': 1.0, 'relu': 0.05, 'linear_fc2': 1.0}.get(op_type, 0.1) * (batch_size / 32)
            temp_impact = {'linear_fc1': 0.01, 'relu': 0.0001, 'linear_fc2': 0.01}.get(op_type, 0.001) * (batch_size / 32)
            
            # Ensure minimum values, avoid zero for small batches
            energy = max(energy, 0.001)
            latency = max(latency, 0.01)
            temp_impact = max(temp_impact, 0.0001)
            
            # print(f"Warning: No profiled data for op_type '{op_type}' batch {batch_size}. Using defaults.")
            return {"energy_joules": energy, "latency_ms": latency, "temp_impact": temp_impact}

# --- Expert Definitions ---

class SimpleExpert(nn.Module):
    """
    A basic Feed-Forward Network acting as an MoE expert.
    This will be used for initial kernel profiling.
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
    """
    Simulated 4-bit Quantized Expert.
    Weights are stored as packed INT8 (representing two 4-bit nibbles).
    Includes explicit (PyTorch-based) dequantization in the forward pass
    to make the dequantization overhead visible to profilers.
    """
    def __init__(self, d_model: int, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.d_model = d_model

        # Simulate 4-bit weights packed into int8
        # Shape: (out_features, in_features / 2) because each byte holds two 4-bit values
        # For fc1: (d_model * 2, d_model / 2)
        self.fc1_weights_packed = nn.Parameter(
            torch.randint(-8, 7, (d_model * 2, d_model // 2), dtype=torch.int8), requires_grad=False
        )
        # Scales are typically float16 in quantization schemes
        self.fc1_scales = nn.Parameter(torch.randn(d_model * 2, 1, dtype=torch.float16), requires_grad=False)
        self.fc1_bias = nn.Parameter(torch.randn(d_model * 2), requires_grad=True)

        # For fc2: (d_model, d_model) -> (d_model, d_model / 2) packed
        self.fc2_weights_packed = nn.Parameter(
            torch.randint(-8, 7, (d_model, d_model), dtype=torch.int8), requires_grad=False
        )
        self.fc2_scales = nn.Parameter(torch.randn(d_model, 1, dtype=torch.float16), requires_grad=False)
        self.fc2_bias = nn.Parameter(torch.randn(d_model), requires_grad=True)

    def _dequantize_and_unpack(self, packed_weights: torch.Tensor, scales: torch.Tensor,
                               d_out: int, d_in: int) -> torch.Tensor:
        """
        Simulates the dequantization and unpacking process on the GPU using PyTorch ops.
        This is a simplified version of the Arm paper's "fast decompression path."
        It will generate underlying CUDA kernels for bitwise ops, shifts, etc.
        """
        # Ensure packed_weights are on the same device as scales and will be processed.
        packed_weights = packed_weights.to(scales.device)

        # 1. Unpack nibbles:
        # Assuming high nibble (bits 4-7) and low nibble (bits 0-3) are packed per byte
        low_nibbles = (packed_weights & 0x0F).to(torch.int8)
        high_nibbles = (packed_weights >> 4).to(torch.int8)

        # 2. Restore signedness (original values were -8 to 7, stored as 0-15)
        # If value > 7, then it's a negative number (e.g., 15 was -1, 8 was -8)
        low_nibbles = torch.where(low_nibbles > 7, low_nibbles - 16, low_nibbles)
        high_nibbles = torch.where(high_nibbles > 7, high_nibbles - 16, high_nibbles)

        # 3. Interleave and reconstruct the full (d_out, d_in) float tensor
        # This is simplified. The paper talks about SIMD-aware packing/interleaving.
        # This reconstruction should mirror how weights were packed offline.
        # For simplicity, let's assume `in_features` is correctly restored from `d_in`.
        # The unpacked tensor will have shape (d_out, d_in).
        unpacked_weights = torch.empty(d_out, d_in, device=packed_weights.device, dtype=scales.dtype)
        # Fill even columns with high_nibbles, odd columns with low_nibbles
        # This assumes a packing scheme where 2 nibbles from one original weight are not in the same byte.
        # If each byte contains 2 separate weights, this is how you'd fill it.
        # This specific interleaving depends on your chosen packing format.
        # A more direct interpretation of Arm paper: if a byte has w0, w16, then w0 goes to col 0, w16 to col 16
        # Here we do a generic interleave.
        unpacked_weights[:, 0::2] = high_nibbles.float() # Assuming high_nibbles correspond to even columns
        unpacked_weights[:, 1::2] = low_nibbles.float()  # Assuming low_nibbles correspond to odd columns

        # 4. Apply scales
        # Scales are (d_out, 1) and will broadcast. Convert scales to unpacked_weights' dtype (float32 assumed for matmul)
        return unpacked_weights * scales.to(unpacked_weights.dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- First Linear Layer (FC1) ---
        # Explicit dequantization step to make its kernels visible
        fc1_weights_dequant = self._dequantize_and_unpack(
            self.fc1_weights_packed, self.fc1_scales,
            self.fc1_weights_packed.shape[0], self.d_model
        )
        x = F.linear(x, fc1_weights_dequant, self.fc1_bias)

        # --- ReLU Activation ---
        x = F.relu(x)

        # --- Second Linear Layer (FC2) ---
        fc2_weights_dequant = self._dequantize_and_unpack(
            self.fc2_weights_packed, self.fc2_scales,
            self.fc2_weights_packed.shape[0], self.d_model * 2 # fc2's input feature dim
        )
        x = F.linear(x, fc2_weights_dequant, self.fc2_bias)
        return x

# --- MoE Layer and Block Definitions ---

class SimpleMoELayer(nn.Module):
    def __init__(self, gate: nn.Module, experts: nn.ModuleList, top_k: int,
                 kernel_cost_model: KernelCostModel, # Added for Phase 2+
                 gpu_system_monitor, # Added for Phase 2+ (forward declaration)
                 routing_strategy: str = "baseline"):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.n_experts = len(experts)
        self.top_k = top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor # Will be used by router
        
        # We need to explicitly pass KernelCostModel and GpuSystemMonitor to the router
        # Router definition will be in routers.py, so a forward declaration/placeholder
        # or import is needed. For now, we assume it's available.
        # From iteration 3 onwards, AdaptiveRouter will use these.
        self.router = AdaptiveRouter(self.n_experts, top_k,
                                     kernel_cost_model, gpu_system_monitor, # Pass the new dependencies
                                     strategy=routing_strategy)
        
        self.expert_cumulative_timings_ms: Dict[int, float] = {} # Total time spent by each expert
        self.metrics_buffer: Dict[str, Any] = {} # To store per-batch metrics for logging

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        num_tokens, d_model = x.shape
        device = x.device

        gate_logits = self.gate(x)
        
        # Pass num_tokens to router (needed for kernel_cost_model lookup based on batch_size)
        top_k_indices, top_k_probs = self.router(gate_logits, num_tokens) 
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
                 gpu_system_monitor, # Added
                 routing_strategy: str = "baseline",
                 expert_type: str = "simple"): # New arg to select expert type
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        self.routing_strategy = routing_strategy
        self.expert_type = expert_type

        self.gate = nn.Linear(d_model, num_experts)

        if expert_type == "simple":
            experts = nn.ModuleList([SimpleExpert(d_model, i) for i in range(num_experts)])
        elif expert_type == "quantized":
            experts = nn.ModuleList([QuantizedExpert(d_model, i) for i in range(num_experts)])
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
            expert_id = int(token_expert_indices[j].item())
            prob = token_expert_probs[j].item() # Probability of this token going to this expert

            # Sum the predicted costs of constituent kernels for this expert
            # We assume a fixed structure: fc1, relu, fc2
            # The d_model here is the *base* d_model for the expert (input to fc1)
            # The KernelCostModel's get_cost method should handle mapping d_model_base to op-specific dimensions.
            
            expert_predicted_energy_cost_for_token = 0.0
            
            # --- IMPORTANT: These op_type strings MUST match what you profile in expert_kernel_profiler.py ---
            # And the d_model values passed to get_cost must align with how you profiled them.
            # get_cost will map d_model_base to the actual op dimensions.
            op_cost_fc1 = kernel_cost_model.get_cost("linear_fc1", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_fc1.get("energy_joules", 0.0)

            op_cost_relu = kernel_cost_model.get_cost("relu", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_relu.get("energy_joules", 0.0)
            
            op_cost_fc2 = kernel_cost_model.get_cost("linear_fc2", effective_kernel_batch_size)
            expert_predicted_energy_cost_for_token += op_cost_fc2.get("energy_joules", 0.0)
            
            # Accumulate weighted predicted energy
            total_predicted_energy += prob * expert_predicted_energy_cost_for_token

    return alpha * total_predicted_energy