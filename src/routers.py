# src/routers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# Forward declarations for type hinting (actual classes defined elsewhere)
class KernelCostModel: pass
class GpuSystemMonitor: pass

class AdaptiveRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int,
                 kernel_cost_model: KernelCostModel,
                 gpu_system_monitor: GpuSystemMonitor,
                 strategy: str = "baseline"):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        self.strategy = strategy # "baseline", "static_optimal", "kernel_aware_ttha"

        # --- TTHA related components (placeholders for Iteration 4) ---
        self.ttha_adapter = None
        self.ttha_optimizer = None
        self.ttha_history = {'power_loss': [], 'temp_loss': [], 'latency_penalty': []}
        self.base_latency_for_penalty = 0.0 # Will be set in main script after baseline runs

        if self.strategy == "kernel_aware_ttha":
            # Initialize TTHA adapter and optimizer here in Iteration 4
            # e.g., self.ttha_adapter = nn.Sequential(...)
            # e.g., self.ttha_optimizer = optim.Adam(...)
            pass


    def forward(self, gate_logits: torch.Tensor, current_batch_size: int):
        """
        Routes tokens to experts based on the selected strategy.
        Args:
            gate_logits: Logits from the gating network (num_tokens, num_experts).
            current_batch_size: The actual number of tokens in the current input batch.
                                This is crucial for looking up kernel costs based on token count.
        """
        device = gate_logits.device
        
        # Predicted costs from KernelCostModel (for kernel_aware strategies)
        # This logic determines 'base_cost_biases' based on pre-profiled data.
        # It's an array of biases, one for each expert.
        base_cost_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)
        if self.strategy in ["static_optimal", "kernel_aware_ttha"]:
            # Iterate through each expert to get its predicted cost for this batch_size
            for expert_id in range(self.num_experts):
                # IMPORTANT: 'current_batch_size' needs to map to the 'batch_size'
                # used during kernel profiling in Phase 1. If your experts process
                # tokens individually even in a batch, then you'd use 1 for lookup.
                # If a whole batch of tokens might hit one expert and run a GEMM with that batch_size,
                # then use current_batch_size.
                # For now, let's assume experts process tokens individually and we're just summing costs per token.
                # So we lookup costs for batch_size=1 (single token through expert's kernels).
                effective_kernel_lookup_batch_size = 1 # Each token passes through an expert as a "batch of 1" for its FFN.
                
                # Sum costs for constituent ops of one expert (e.g., fc1, relu, fc2)
                expert_predicted_energy_cost = 0.0
                expert_predicted_temp_impact = 0.0

                # --- These op_type strings MUST match what you profile in expert_kernel_profiler.py ---
                # And the d_model (implicitly handled by KernelCostModel's get_cost based on op_type)
                op_cost_fc1 = self.kernel_cost_model.get_cost("linear_fc1", effective_kernel_lookup_batch_size)
                expert_predicted_energy_cost += op_cost_fc1.get("energy_joules", 0.0)
                expert_predicted_temp_impact += op_cost_fc1.get("temp_impact", 0.0)

                op_cost_relu = self.kernel_cost_model.get_cost("relu", effective_kernel_lookup_batch_size)
                expert_predicted_energy_cost += op_cost_relu.get("energy_joules", 0.0)
                expert_predicted_temp_impact += op_cost_relu.get("temp_impact", 0.0)
                
                op_cost_fc2 = self.kernel_cost_model.get_cost("linear_fc2", effective_kernel_lookup_batch_size)
                expert_predicted_energy_cost += op_cost_fc2.get("energy_joules", 0.0)
                expert_predicted_temp_impact += op_cost_fc2.get("temp_impact", 0.0)

                # Convert predicted costs into a bias for gate_logits
                # Higher energy/temp should lead to more negative bias (less likely to be chosen)
                # These coefficients (e.g., 100, 50) are hyperparameters you'll tune.
                base_cost_biases[expert_id] = -(expert_predicted_energy_cost * 100.0 + expert_predicted_temp_impact * 50.0)

        # Apply strategy-specific logic
        final_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)

        if self.strategy == "baseline":
            # Baseline: No additional bias, just standard MoE routing
            final_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)
        elif self.strategy == "static_optimal":
            # Static optimal: Uses the pre-computed base_cost_biases
            final_biases = base_cost_biases # No dynamic adjustment here
        elif self.strategy == "kernel_aware_ttha":
            # TTHA Logic (Iteration 4): Combines base_cost_biases with dynamic biases from TTHA adapter
            if self.ttha_adapter is None:
                # Fallback if TTHA not initialized (shouldn't happen in a proper setup)
                final_biases = base_cost_biases
            else:
                # 1. Get current overall GPU stats
                gpu_stats = self.gpu_system_monitor.get_current_stats()
                current_gpu_temp = torch.tensor(gpu_stats['temperature'], dtype=gate_logits.dtype, device=device).unsqueeze(0)
                current_gpu_power = torch.tensor(gpu_stats['power_watt'], dtype=gate_logits.dtype, device=device).unsqueeze(0)

                # Input to TTHA adapter: Flattened predicted costs + current global GPU stats
                # This input needs to match the dimension of ttha_adapter's first layer.
                # Here, assuming 2 * num_experts (for energy, temp for each) + 2 (global temp, power)
                ttha_input = torch.cat([predicted_costs_tensor.flatten(), current_gpu_temp, current_gpu_power])
                
                # Output: dynamic biases generated by the small TTHA MLP
                dynamic_biases = self.ttha_adapter(ttha_input.unsqueeze(0)).squeeze(0) # (1, input_dim) -> (1, num_experts) -> (num_experts,)
                
                # Combine base biases from pre-profiled kernel costs with TTHA dynamic biases
                final_biases = base_cost_biases + dynamic_biases

        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

        biased_logits = gate_logits + final_biases
        topk_vals, topk_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1)
        
        # Note: If you implement an aux_loss in the router, it should be calculated here
        # and returned if needed for outer loss function. For now, it's in MoELayer.

        return topk_indices, routing_weights

    # --- TTHA Update Method (for Iteration 4) ---
    def update_ttha(self, observed_metrics: Dict[str, float],
                    target_power: float = 100.0, target_temp: float = 70.0,
                    latency_penalty_weight: float = 0.05):
        """
        Performs a lightweight update to the TTHA adapter based on observed hardware metrics.
        This method will be populated in Iteration 4.
        """
        if self.strategy != "kernel_aware_ttha" or self.ttha_adapter is None:
            return

        # TTHA update logic will go here in Iteration 4
        # Example:
        # observed_power = observed_metrics.get("gpu_power_watt", 0.0)
        # observed_temp = observed_metrics.get("gpu_temperature_c", 0.0)
        # observed_latency = observed_metrics.get("inference_latency_ms", 0.0)
        # ttha_total_loss = ... calculate loss based on observed vs target ...
        # self.ttha_optimizer.zero_grad()
        # ttha_loss_tensor = torch.tensor(ttha_total_loss, device=self.ttha_adapter[0].weight.device, dtype=torch.float32)
        # ttha_loss_tensor.backward()
        # self.ttha_optimizer.step()
        # self.ttha_history['power_loss'].append(...)
        pass