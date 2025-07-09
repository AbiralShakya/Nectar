# src/moe_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, Tuple, Any, Optional, List, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict

# Import NECTAR components (GpuSystemMonitor, KernelCostModel are fine at top level)
from src.monitor import GpuSystemMonitor
from src.kernelcostmodel import KernelCostModel

@dataclass
class MoEConfig:
    """Configuration for MoE models with realistic defaults"""
    d_model: int = 4096
    num_experts: int = 8
    top_k: int = 2
    expert_capacity_factor: float = 1.25  # For capacity-based routing
    dropout: float = 0.1
    use_bias: bool = False
    activation: str = "swiglu"  # Modern activation for LLMs
    expert_type: str = "swiglu_ffn"  # More realistic expert type
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    expert_dropout: float = 0.0
    use_grouped_gemm: bool = True  # For efficient batched expert computation
    capacity_factor: float = 1.0  # Token capacity per expert
    batch_size: int = 64 # Added batch_size here
    
    # --- LaCT-specific parameters ---
    lact_chunk_size: int = 2048 # Number of tokens after which internal expert TTT update occurs
    lact_lr: float = 1e-3 # Learning rate for internal fast weights
    lact_fast_weight_dim_ratio: float = 0.25 # Ratio of fast weight hidden dim to d_model (e.g., 0.25 * d_model)
    lact_update_frequency_tokens: int = 1000 # Default to a smaller, more frequent update for initial testing
    quantization_bits: int = 8 # Moved here for global config access (used by OptimizedQuantizedExpert)


class SwiGLUExpert(nn.Module):
    """
    Modern SwiGLU-based expert matching what's used in production LLMs.
    More efficient and realistic than simple ReLU-based experts.
    """
    def __init__(self, config: MoEConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.config = config
        
        # SwiGLU uses 3 linear layers: gate, up, down
        # Hidden dimension is typically 8/3 * d_model for SwiGLU to match parameter count
        self.hidden_dim = int(8 * config.d_model / 3)
        
        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.use_bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.expert_dropout) if config.expert_dropout > 0 else None
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for stability"""
        std = math.sqrt(2.0 / self.config.d_model)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std / math.sqrt(self.hidden_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU forward pass: SwiGLU(x) = (Swish(gate(x)) * up(x)) @ down
        """
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # SwiGLU activation: Swish(gate) * up
        swish_gate = F.silu(gate_output)  # SiLU is the same as Swish
        activated = swish_gate * up_output
        
        if self.dropout is not None:
            activated = self.dropout(activated)
        
        return self.down_proj(activated)


class OptimizedQuantizedExpert(nn.Module):
    """
    Improved quantized expert with better packing and realistic quantization schemes.
    Uses INT8 quantization which is more practical than 2/4-bit for current hardware.
    """
    def __init__(self, config: MoEConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.config = config
        self.quantization_bits = config.quantization_bits # Access quantization_bits from config
        self.hidden_dim = int(8 * config.d_model / 3)
        
        # For actual INT8 quantized inference, these would be nn.quantized.Linear
        # or custom Triton kernels for W8A16.
        # For this prototype, we'll use regular Linear layers and assume quantization
        # happens either implicitly or via pre-processing, allowing profiling of "quantized" ops.
        self.gate_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias)
        self.up_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias)
        self.down_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.use_bias)
        
        # Ensure initial weights are in float if we're not doing explicit quantization here
        self.gate_proj.weight.data = self.gate_proj.weight.data.half()
        self.up_proj.weight.data = self.up_proj.weight.data.half()
        self.down_proj.weight.data = self.down_proj.weight.data.half()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized operations (simulated by using half precision here)"""
        original_dtype = x.dtype # Store the original dtype

        # In a real W8A16 setup, x might be FP16/BF16, and operations would run on quantized weights.
        # Here, we cast x to half for a simplified "quantized" path.
        x = x.half()
        
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        swish_gate = F.silu(gate_output)
        activated = swish_gate * up_output
        
        output = self.down_proj(activated)
        return output.to(original_dtype) # Return in original dtype


# --- NEW: SwiGLUFastWeightNet (Internal Fast Weights for LaCT Expert) ---
class SwiGLUFastWeightNet(nn.Module):
    """
    A small, non-linear SwiGLU MLP that serves as the "fast weights" network
    within a LaCTMoEExpert. Its parameters are updated during inference.
    Corresponds to fW(x) in LaCT paper
    """
    def __init__(self, d_model: int, fast_weight_dim: int):
        super().__init__()
        self.d_model = d_model
        self.fast_weight_dim = fast_weight_dim
        
        # Parameters for the fast weight network (typically smaller than main model)
        # LaCT uses W1, W2, W3 (SwiGLU-MLP)
        # Consistent with F.linear(input, weight) where weight is [out_features, in_features]
        # x: [B, d_model], w1: [fast_weight_dim, d_model] -> F.linear(x, w1)
        self.w1 = nn.Parameter(torch.randn(fast_weight_dim, d_model)) # Changed shape here
        self.w3 = nn.Parameter(torch.randn(fast_weight_dim, d_model)) # Changed shape here
        
        # activated: [B, fast_weight_dim], w2: [d_model, fast_weight_dim] -> F.linear(activated, w2)
        self.w2 = nn.Parameter(torch.randn(d_model, fast_weight_dim)) # Changed shape here
        
        # Initialize weights
        # Adjusting std for new shapes
        std_w1_w3 = math.sqrt(2.0 / d_model)
        std_w2 = math.sqrt(2.0 / fast_weight_dim) # w2's fan-in is fast_weight_dim

        nn.init.normal_(self.w1, mean=0.0, std=std_w1_w3)
        nn.init.normal_(self.w3, mean=0.0, std=std_w1_w3)
        nn.init.normal_(self.w2, mean=0.0, std=std_w2)
        
        # Fast weights are typically not part of the main model's optimization graph
        # and should only be updated by their specific optimizer within LaCTMoEExpert
        self.w1.requires_grad_(False) 
        self.w2.requires_grad_(False)
        self.w3.requires_grad_(False)
    
    def apply_fw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the fast weight network to query (Q) tokens.
        Corresponds to LaCT's apply_fw(fast_weight, q)
        """
        # x is [batch_size_chunk, d_model]
        # F.linear(input, weight) is equivalent to input @ weight.T
        hidden_gate = F.linear(x, self.w1) # [B, fast_weight_dim]
        hidden_up = F.linear(x, self.w3)   # [B, fast_weight_dim]
        
        activated = F.silu(hidden_gate) * hidden_up # [B, fast_weight_dim]
        
        output = F.linear(activated, self.w2) # [B, d_model]
        return output

    def compute_update_gradients(self, k: torch.Tensor, v: torch.Tensor, lr_coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes gradients for fast weights based on a chunk of K and V.
        Corresponds to the gradient computation inside update() in LaCT's Alg. 1.
        Returns a dict of gradients for w1, w2, w3.
        """
        # Temporarily enable gradients for fast weights for this update step
        self.w1.requires_grad_(True)
        self.w2.requires_grad_(True)
        self.w3.requires_grad_(True)

        # Compute loss with k and v
        # LfW(k,v) = -fW(k)^T v (Negative dot product loss from LaCT Eq. 7)
        # k and v are [batch_size_chunk, d_model]
        
        # Forward pass through fast weight net with k
        gate_before_act = F.linear(k, self.w1)
        hidden_before_gate = F.linear(k, self.w3)
        hidden = F.silu(gate_before_act) * hidden_before_gate
        
        fW_k = F.linear(hidden, self.w2) # [batch_size_chunk, d_model]
        
        # Negative dot product loss, summed over d_model dimension
        # (fW_k * v) is [batch_size_chunk, d_model]
        fast_weight_loss_per_token = -(fW_k * v).sum(dim=-1) # [batch_size_chunk]
        
        # Apply learning rate coefficients per token if provided (e.g., from LaCT's lr input)
        # For simplicity, apply a single LR to the mean loss for now
        if lr_coeffs is not None and lr_coeffs.ndim == 2: # [B, 3] for w1,w2,w3 LRs or just [B,1] for scalar
             # Apply element-wise learning rate to each token's loss before summing/meaning
             fast_weight_loss_per_token = fast_weight_loss_per_token * lr_coeffs[:, 0] # Use first LR coeff

        # Sum loss over the chunk for gradient computation
        chunk_loss = fast_weight_loss_per_token.mean() # Mean over chunk, then backward
        
        # Compute gradients for fast weights
        # retain_graph=False is appropriate here as it's a self-contained update
        grads = torch.autograd.grad(chunk_loss, [self.w1, self.w2, self.w3], retain_graph=False)
        
        # Return gradients as a dict, mapping to param names
        # Ensure correct mapping: grads[0] for w1, grads[1] for w2, grads[2] for w3
        return {'w1': grads[0], 'w2': grads[1], 'w3': grads[2]}

# --- NEW: LaCTMoEExpert ---
class LaCTMoEExpert(nn.Module):
    """
    An MoE expert that incorporates Test-Time Training (TTT) principles from LaCT.
    It has its own internal "fast weights" that are updated during inference.
    """
    def __init__(self, config: MoEConfig, expert_id: int):
        super().__init__()
        self.expert_id = expert_id
        self.config = config
        
        # 1. Internal Fast Weights Network
        # fast_weight_dim scales with d_model (e.g., 0.25 * d_model)
        fast_weight_dim = int(config.d_model * config.lact_fast_weight_dim_ratio)
        self.fast_weight_net = SwiGLUFastWeightNet(config.d_model, fast_weight_dim)
        
        # 2. Optimizer for Fast Weights (AdamW for now, Muon later)
        # Only the fast_weight_net's parameters are passed here
        self.fast_weight_optimizer = torch.optim.AdamW(
            self.fast_weight_net.parameters(), 
            lr=config.lact_lr,
            betas=(0.9, 0.999), 
            eps=1e-8, 
            weight_decay=1e-5 # Small weight decay for fast weights
        )
        # Scheduler for fast weights. T_max in terms of update steps.
        self.fast_weight_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.fast_weight_optimizer, T_max=1000 # Example T_max for internal updates
        )
        
        # 3. Internal State for Chunk-wise Updates
        self.chunk_buffer_k = []   # To store keys for fast weight update loss
        self.chunk_buffer_v = []   # To store values for fast weight update loss
        self.chunk_buffer_lr_coeffs = [] # For per-token LR, if applicable
        self.current_chunk_tokens_count = 0 # Tracks tokens processed in current chunk

        self.lact_update_chunk_size = config.lact_chunk_size # Number of tokens per chunk for updates
        self.lact_update_frequency_tokens = config.lact_update_frequency_tokens # Tokens seen before allowing an update
        self.tokens_since_last_update = 0 # Counter for `lact_update_frequency_tokens`

    def forward(self, x: torch.Tensor, router_metadata: Dict[str, Any] = None) -> torch.Tensor:
        """
        Forward pass for LaCTMoEExpert. Applies fast weights and buffers tokens for chunk-wise update.
        
        Args:
            x: Input tokens to this expert [num_tokens_routed_to_expert, d_model]
            router_metadata: Dict from AdaptiveRouter, possibly containing control signals for LaCT updates.
        """
        device = x.device
        num_tokens_in_current_batch = x.size(0)

        # 1. Apply Fast Weights Network (fW(Q))
        # This is the "apply" operation.
        expert_output = self.fast_weight_net.apply_fw(x) # [num_tokens_routed_to_expert, d_model]

        # 2. Buffer Tokens and Context for Chunk-wise Update
        # For simplicity, K and V for update are derived directly from X.
        # In a full Transformer, these would be proper Q, K, V projections from an attention layer.
        k_for_update = x.detach() # Detach to prevent gradients flowing back into MoE input
        v_for_update = x.detach() 
        # Dummy lr_coeffs for now, in LaCT these are learned/predicted per token
        lr_coeffs_for_update = torch.ones(num_tokens_in_current_batch, 3, device=device) 

        self.chunk_buffer_k.append(k_for_update)
        self.chunk_buffer_v.append(v_for_update)
        self.chunk_buffer_lr_coeffs.append(lr_coeffs_for_update)
        self.current_chunk_tokens_count += num_tokens_in_current_batch
        self.tokens_since_last_update += num_tokens_in_current_batch

        # 3. Perform Chunk-wise Fast Weight Update (Update operation)
        # Trigger update if chunk size reached AND enough tokens passed since last update
        perform_update = False
        if self.current_chunk_tokens_count >= self.lact_update_chunk_size:
            if self.tokens_since_last_update >= self.lact_update_frequency_tokens:
                perform_update = True
        
        # NECTAR's control override: If router_metadata has a 'force_lact_update' signal
        if router_metadata and router_metadata.get('force_lact_update', False):
            perform_update = True 
        
        # NECTAR's control override: Dynamic chunk size from router_metadata
        # The router could pass 'dynamic_lact_chunk_size'
        if router_metadata and 'dynamic_lact_chunk_size' in router_metadata:
            self.lact_update_chunk_size = router_metadata['dynamic_lact_chunk_size']


        if perform_update and self.chunk_buffer_k: # Only update if buffer is not empty
            self._perform_lact_update()
            # Reset chunk buffers after update
            self.current_chunk_tokens_count = 0
            self.chunk_buffer_k = []
            self.chunk_buffer_v = []
            self.chunk_buffer_lr_coeffs = []
            self.tokens_since_last_update = 0 # Reset counter after update

        return expert_output

    def _perform_lact_update(self):
        """
        Aggregates buffered tokens into a chunk and performs a single
        fast weight update using LaCT's principles.
        Corresponds to update() in LaCT's Alg. 1
        """
        # Concatenate buffered data to form a single chunk for update
        chunk_k = torch.cat(self.chunk_buffer_k, dim=0)
        chunk_v = torch.cat(self.chunk_buffer_v, dim=0)
        chunk_lr_coeffs = torch.cat(self.chunk_buffer_lr_coeffs, dim=0)
        
        self.fast_weight_optimizer.zero_grad()
        
        # Compute gradients for fast weights
        fast_weight_grads = self.fast_weight_net.compute_update_gradients(
            chunk_k, chunk_v, chunk_lr_coeffs
        )
        
        # Apply gradients to fast weights (Optimizer step)
        # Manually apply gradients by setting .grad attribute (as autograd.grad was used)
        for param_name, grad_tensor in fast_weight_grads.items():
            param = None
            if param_name == 'w1': param = self.fast_weight_net.w1
            elif param_name == 'w2': param = self.fast_weight_net.w2
            elif param_name == 'w3': param = self.fast_weight_net.w3

            if param is not None:
                if param.grad is not None: # Ensure previous grad is cleared
                    param.grad.data.zero_()
                param.grad = grad_tensor # Assign the computed gradient
        
        # Perform optimizer step and scheduler step
        self.fast_weight_optimizer.step()
        self.fast_weight_scheduler.step()
        
        # Apply L2 Weight Normalization (as in LaCT's update)
        with torch.no_grad():
            for param in self.fast_weight_net.parameters():
                if param.dim() > 1: # Apply L2-norm along input dimension (dim 0 for weights)
                    # Normalize along rows (dim=0 for nn.Parameter (column-major) or input dim for weight matrix)
                    # LaCT applies L2-Normalize(W - g), then (W-g).norm(dim=1).
                    # Simplified: Re-normalize after update.
                    param.data = F.normalize(param.data, p=2, dim=0) 
        
        # Disable requires_grad for fast weights after update to prevent accidental tracking outside this function
        # They should only be trainable within this specific update step.
        self.fast_weight_net.w1.requires_grad_(False)
        self.fast_weight_net.w2.requires_grad_(False)
        self.fast_weight_net.w3.requires_grad_(False)

def compute_energy_loss(selected_expert_indices: torch.Tensor, expert_profiles: Dict[str, Dict], alpha=0.001):
    energy = 0.0
    for idx in selected_expert_indices.view(-1):
        profile = expert_profiles.get(str(int(idx.item())))
        if profile:
            energy += profile.get("energy_cost", 0.0)
    return alpha * energy


class CapacityBasedRouter(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=config.use_bias) # Use config.use_bias
        self.expert_capacity = None  # Will be set dynamically
        
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size_seq, d_model = x.shape
        
        router_logits = self.gate(x)
        temperature = 1.0
        router_logits = router_logits / temperature
        
        router_probs = F.softmax(router_logits, dim=-1)
        
        top_k_logits, top_k_indices = torch.topk(router_logits, self.config.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Normalize probabilities (important for training stability)
        # Sum of top-k probabilities for a token should be 1.0
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8) # Add clamp for stability
        
        if self.expert_capacity is None:
            tokens_per_expert = batch_size_seq / self.config.num_experts
            self.expert_capacity = int(tokens_per_expert * self.config.capacity_factor)
        
        top_k_indices, top_k_probs = self._apply_capacity_constraints(
            top_k_indices, top_k_probs, router_probs
        )
        
        aux_losses = self._compute_aux_losses(router_logits, router_probs, top_k_indices)
        
        return top_k_indices, top_k_probs, aux_losses
    
    def _apply_capacity_constraints(self, expert_indices, expert_probs, router_probs):
        num_tokens = expert_indices.shape[0]
        
        # Prepare for counting
        one_hot_indices = F.one_hot(expert_indices.view(-1), num_classes=self.config.num_experts).float()
        # Sum over tokens to get count for each expert
        expert_counts = one_hot_indices.sum(dim=0)
        
        # Mask out tokens that exceed capacity (simplified logic)
        # This implementation requires more careful token reassignment if multiple experts are over capacity
        # For simplicity, this acts as a hard cap on expert_counts after initial assignment.
        # A more advanced capacity algorithm (e.g., in Switch Transformers) would re-route excess tokens.
        # Here, tokens exceeding capacity will be "dropped" from expert_probs.
        
        # Calculate assignments per expert per token. expert_mask_flat is [num_tokens * top_k]
        expert_mask_flat = expert_indices.view(-1)
        expert_probs_flat = expert_probs.view(-1)
        
        # Store initial indices/probs for modification
        modified_expert_indices = expert_indices.clone()
        modified_expert_probs = expert_probs.clone()

        for expert_id in torch.where(expert_counts > self.expert_capacity)[0]: # Iterate only over over-capacity experts
            expert_tokens_mask = (expert_indices == expert_id) # [num_tokens, top_k] mask for this expert
            
            # Get probabilities for tokens assigned to this specific expert
            probs_for_expert = expert_probs[expert_tokens_mask] # Flattened list of probs for this expert
            
            if probs_for_expert.numel() > self.expert_capacity:
                # Get the indices of the highest probability assignments
                _, top_probs_local_indices = torch.topk(probs_for_expert, self.expert_capacity, largest=True)
                
                # Create a mask for tokens to keep *within this expert's assignments*
                keep_mask_for_expert_assignments = torch.zeros_like(probs_for_expert, dtype=torch.bool)
                keep_mask_for_expert_assignments[top_probs_local_indices] = True
                
                # Get the global indices (row_idx, col_idx) for tokens to drop
                global_indices_to_this_expert = torch.where(expert_tokens_mask) # (token_row_indices, top_k_col_indices)
                tokens_to_drop_global_row_idx = global_indices_to_this_expert[0][~keep_mask_for_expert_assignments]
                tokens_to_drop_global_col_idx = global_indices_to_this_expert[1][~keep_mask_for_expert_assignments]
                
                # Apply changes to the cloned tensors
                modified_expert_probs[tokens_to_drop_global_row_idx, tokens_to_drop_global_col_idx] = 0.0
                modified_expert_indices[tokens_to_drop_global_row_idx, tokens_to_drop_global_col_idx] = -1 # Mark as dropped
        
        # Final filtering for any token with all assignments dropped
        final_expert_probs = modified_expert_probs
        final_expert_indices = modified_expert_indices
        
        return final_expert_indices, final_expert_probs
    
    def _compute_aux_losses(self, router_logits, router_probs, expert_indices):
        num_tokens = router_logits.shape[0]
        
        # Load balancing loss (encourage uniform expert usage)
        # Use F.one_hot on selected top-1 expert for proper load balancing based on assigned token counts
        # Filter out dropped tokens for top-1 usage calculation
        valid_top1_expert_indices = expert_indices[:, 0] # Get top-1 expert for each token
        valid_top1_expert_indices = valid_top1_expert_indices[valid_top1_expert_indices != -1] # Filter dropped tokens

        if valid_top1_expert_indices.numel() == 0:
            tokens_per_expert_top1 = torch.zeros(self.config.num_experts, device=router_logits.device)
        else:
            # Count how many valid top-1 assignments each expert received
            tokens_per_expert_top1 = F.one_hot(valid_top1_expert_indices, num_classes=self.config.num_experts).sum(dim=0).float()
        
        # Sum of probabilities of router for experts (from original router_probs, not modified)
        sum_router_probs_per_expert = router_probs.sum(dim=0) # [num_experts]
        
        # Aux loss as defined in original MoE papers (e.g., Switch Transformers Eq. 3)
        # (sum of expert router probabilities * sum of tokens routed to expert)
        load_balance_loss = (sum_router_probs_per_expert * tokens_per_expert_top1).sum()
        # Normalize by (total_tokens * num_experts)
        load_balance_loss = load_balance_loss * self.config.load_balance_weight / (num_tokens * self.config.num_experts)

        # Router z-loss (encourage router to be decisive, avoid very small logits)
        # logsumexp(logits) should not be too far from max(logits)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        router_z_loss = router_z_loss * self.config.router_z_loss_weight
        
        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "expert_usage": tokens_per_expert_top1, # Actual counts for logging (from valid top-1)
        }


class NetworkTopologyOptimizer:
    """
    Optimizes expert placement and data movement across GPU cluster.
    Reduces inter-GPU communication and balances load.
    """
    def __init__(self, num_gpus: int, num_experts: int, 
                 bandwidth_matrix: Optional[torch.Tensor] = None,
                 latency_matrix: Optional[torch.Tensor] = None):
        self.num_gpus = num_gpus
        self.num_experts = num_experts
        self.experts_per_gpu = num_experts // num_gpus
        
        # Network topology matrices (if not provided, assume uniform)
        if bandwidth_matrix is None:
            self.bandwidth_matrix = torch.ones(num_gpus, num_gpus) * 100.0  # GB/s
        else:
            self.bandwidth_matrix = bandwidth_matrix
            
        if latency_matrix is None:
            self.latency_matrix = torch.ones(num_gpus, num_gpus) * 0.001  # 1ms
        else:
            self.latency_matrix = latency_matrix
        
        # Expert placement strategy
        self.placement_strategy = "load_balanced"  # "load_balanced", "bandwidth_optimized", "thermal_aware"
        
        # Track expert usage patterns
        self.expert_usage_history = defaultdict(list)
        self.gpu_load_history = defaultdict(list)
        
    def optimize_expert_placement(self, expert_usage_stats: Dict[int, float],
                                gpu_temps: List[float],
                                gpu_memory_usage: List[float]) -> Dict[int, int]:
        """
        Optimize expert placement based on usage patterns and hardware state.
        Returns mapping from expert_id to gpu_id.
        """
        if self.placement_strategy == "load_balanced":
            return self._load_balanced_placement(expert_usage_stats, gpu_temps, gpu_memory_usage)
        elif self.placement_strategy == "bandwidth_optimized":
            return self._bandwidth_optimized_placement(expert_usage_stats)
        elif self.placement_strategy == "thermal_aware":
            return self._thermal_aware_placement(expert_usage_stats, gpu_temps)
        else:
            return self._default_placement()
    
    def _load_balanced_placement(self, expert_usage_stats: Dict[int, float],
                               gpu_temps: List[float],
                               gpu_memory_usage: List[float]) -> Dict[int, int]:
        """Place experts to balance load across GPUs."""
        # Sort experts by usage (descending)
        sorted_experts = sorted(expert_usage_stats.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate GPU capacity scores (lower temp and memory usage = higher capacity)
        gpu_capacities = []
        for i in range(self.num_gpus):
            temp_score = max(0, 1.0 - (gpu_temps[i] - 30) / 60)  # Normalize temp to [0,1]
            memory_score = max(0, 1.0 - gpu_memory_usage[i])
            capacity = temp_score * 0.6 + memory_score * 0.4
            gpu_capacities.append(capacity)
        
        # Place experts on GPUs with highest capacity
        placement = {}
        gpu_loads = [0.0] * self.num_gpus
        
        for expert_id, usage in sorted_experts:
            # Find GPU with lowest load relative to capacity
            best_gpu = 0
            best_score = float('inf')
            
            for gpu_id in range(self.num_gpus):
                load_ratio = gpu_loads[gpu_id] / (gpu_capacities[gpu_id] + 1e-8)
                if load_ratio < best_score:
                    best_score = load_ratio
                    best_gpu = gpu_id
            
            placement[expert_id] = best_gpu
            gpu_loads[best_gpu] += usage
        
        return placement
    
    def _bandwidth_optimized_placement(self, expert_usage_stats: Dict[int, float]) -> Dict[int, int]:
        """Place experts to minimize inter-GPU communication."""
        # Group experts that are frequently used together
        expert_groups = self._identify_expert_groups(expert_usage_stats)
        
        placement = {}
        gpu_loads = [0.0] * self.num_gpus
        
        for group in expert_groups:
            # Place entire group on same GPU if possible
            target_gpu = self._find_best_gpu_for_group(group, gpu_loads)
            
            for expert_id in group:
                placement[expert_id] = target_gpu
                gpu_loads[target_gpu] += expert_usage_stats.get(expert_id, 0.0)
        
        return placement
    
    def _thermal_aware_placement(self, expert_usage_stats: Dict[int, float],
                               gpu_temps: List[float]) -> Dict[int, int]:
        """Place experts considering thermal constraints."""
        # Sort GPUs by temperature (ascending)
        gpu_temp_indices = sorted(range(self.num_gpus), key=lambda i: gpu_temps[i])
        
        # Sort experts by usage (descending)
        sorted_experts = sorted(expert_usage_stats.items(), key=lambda x: x[1], reverse=True)
        
        placement = {}
        gpu_loads = [0.0] * self.num_gpus
        
        for expert_id, usage in sorted_experts:
            # Place high-usage experts on cooler GPUs
            for gpu_id in gpu_temp_indices:
                if gpu_loads[gpu_id] < 1.0:  # Assume max load of 1.0
                    placement[expert_id] = gpu_id
                    gpu_loads[gpu_id] += usage
                    break
        
        return placement
    
    def _identify_expert_groups(self, expert_usage_stats: Dict[int, float]) -> List[List[int]]:
        """Identify groups of experts that are frequently used together."""
        # Simple heuristic: group experts by usage patterns
        # In practice, this would use correlation analysis of usage patterns
        groups = []
        used_experts = set()
        
        for expert_id in range(self.num_experts):
            if expert_id in used_experts:
                continue
                
            # Create group with this expert and nearby experts
            group = [expert_id]
            used_experts.add(expert_id)
            
            # Add nearby experts (simple heuristic)
            for other_id in range(expert_id + 1, min(expert_id + 4, self.num_experts)):
                if other_id not in used_experts:
                    group.append(other_id)
                    used_experts.add(other_id)
            
            groups.append(group)
        
        return groups
    
    def _find_best_gpu_for_group(self, expert_group: List[int], gpu_loads: List[float]) -> int:
        """Find the best GPU for placing a group of experts."""
        best_gpu = 0
        best_score = float('inf')
        
        for gpu_id in range(self.num_gpus):
            # Score based on current load and bandwidth to other GPUs
            load_score = gpu_loads[gpu_id]
            bandwidth_score = self._calculate_bandwidth_score(gpu_id, expert_group)
            
            total_score = load_score + bandwidth_score * 0.1
            if total_score < best_score:
                best_score = total_score
                best_gpu = gpu_id
        
        return best_gpu
    
    def _calculate_bandwidth_score(self, gpu_id: int, expert_group: List[int]) -> float:
        """Calculate bandwidth score for placing experts on this GPU."""
        # Lower score = better bandwidth
        total_score = 0.0
        
        for other_gpu in range(self.num_gpus):
            if other_gpu != gpu_id:
                # Penalize low bandwidth connections
                bandwidth = self.bandwidth_matrix[gpu_id, other_gpu]
                total_score += 1.0 / (bandwidth + 1e-8)
        
        return total_score
    
    def _default_placement(self) -> Dict[int, int]:
        """Default round-robin placement."""
        placement = {}
        for expert_id in range(self.num_experts):
            placement[expert_id] = expert_id % self.num_gpus
        return placement
    
    def update_usage_stats(self, expert_usage: Dict[int, int], gpu_loads: List[float]):
        """Update usage statistics for optimization."""
        for expert_id, usage in expert_usage.items():
            self.expert_usage_history[expert_id].append(usage)
            # Keep only recent history
            if len(self.expert_usage_history[expert_id]) > 100:
                self.expert_usage_history[expert_id] = self.expert_usage_history[expert_id][-50:]
        
        for gpu_id, load in enumerate(gpu_loads):
            self.gpu_load_history[gpu_id].append(load)
            if len(self.gpu_load_history[gpu_id]) > 100:
                self.gpu_load_history[gpu_id] = self.gpu_load_history[gpu_id][-50:]
    
    def get_communication_cost(self, expert_placement: Dict[int, int], 
                             token_routing: Dict[int, List[int]]) -> float:
        """Calculate communication cost for given placement and routing."""
        total_cost = 0.0
        
        for token_id, expert_ids in token_routing.items():
            # Find which GPUs these experts are on
            gpu_ids = set(expert_placement[expert_id] for expert_id in expert_ids)
            
            if len(gpu_ids) > 1:
                # Cross-GPU communication needed
                for gpu1 in gpu_ids:
                    for gpu2 in gpu_ids:
                        if gpu1 != gpu2:
                            # Add bandwidth and latency cost
                            bandwidth_cost = 1.0 / self.bandwidth_matrix[gpu1, gpu2]
                            latency_cost = self.latency_matrix[gpu1, gpu2]
                            total_cost += bandwidth_cost + latency_cost
        
        return total_cost

class DistributedMoELayer(nn.Module):
    """
    Distributed MoE layer with network topology optimization.
    Reduces data movement and balances load across GPUs.
    """
    def __init__(self, d_model: int, num_experts: int, num_gpus: int,
                 router_class, expert_class, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.top_k = top_k
        
        # Network topology optimizer
        self.topology_optimizer = NetworkTopologyOptimizer(num_gpus, num_experts)
        
        # Expert placement (expert_id -> gpu_id)
        self.expert_placement = self.topology_optimizer._default_placement()
        
        # Create experts distributed across GPUs
        self.experts = nn.ModuleList()
        for expert_id in range(num_experts):
            gpu_id = self.expert_placement[expert_id]
            expert = expert_class(d_model)
            # In real implementation, would move expert to specific GPU
            self.experts.append(expert)
        
        # Router (replicated on each GPU)
        self.router = router_class(d_model, num_experts, top_k)
        
        # Communication buffers
        self.communication_buffers = {}
        
    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions
        expert_indices, expert_weights, router_metadata = self.router(x, ttt_context)
        
        # Optimize expert placement if needed
        if self._should_reoptimize_placement():
            self._reoptimize_placement(router_metadata)
        
        # Process tokens with minimal data movement
        output = torch.zeros_like(x)
        
        # Group tokens by their target experts to minimize communication
        expert_token_groups = self._group_tokens_by_experts(expert_indices, expert_weights)
        
        for expert_id, token_data in expert_token_groups.items():
            gpu_id = self.expert_placement[expert_id]
            tokens, weights = token_data
            
            if len(tokens) > 0:
                # Process tokens for this expert
                expert_output = self.experts[expert_id](tokens)
                
                # Weight and scatter back to original positions
                weighted_output = expert_output * weights.unsqueeze(-1)
                output.scatter_add_(0, tokens.unsqueeze(-1).expand(-1, d_model), weighted_output)
        
        return output
    
    def _should_reoptimize_placement(self) -> bool:
        """Check if expert placement should be reoptimized."""
        # Reoptimize every 1000 forward passes
        return hasattr(self, '_forward_count') and self._forward_count % 1000 == 0
    
    def _reoptimize_placement(self, router_metadata: Dict[str, Any]):
        """Reoptimize expert placement based on current usage patterns."""
        if 'expert_usage' in router_metadata:
            expert_usage = router_metadata['expert_usage']
            
            # Convert to usage statistics
            usage_stats = {}
            for expert_id, usage in enumerate(expert_usage):
                usage_stats[expert_id] = float(usage)
            
            # Get current GPU state (would come from monitoring)
            gpu_temps = [50.0] * self.num_gpus  # Placeholder
            gpu_memory = [0.5] * self.num_gpus   # Placeholder
            
            # Optimize placement
            new_placement = self.topology_optimizer.optimize_expert_placement(
                usage_stats, gpu_temps, gpu_memory
            )
            
            # Update placement (in real implementation, would migrate experts)
            self.expert_placement = new_placement
    
    def _group_tokens_by_experts(self, expert_indices: torch.Tensor, 
                                expert_weights: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Group tokens by their target experts to minimize communication."""
        batch_size, seq_len, top_k = expert_indices.shape
        
        expert_groups = defaultdict(lambda: ([], []))
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(top_k):
                    expert_id = expert_indices[b, s, k].item()
                    weight = expert_weights[b, s, k].item()
                    
                    if weight > 0.01:  # Only consider significant weights
                        token_idx = b * seq_len + s
                        expert_groups[expert_id][0].append(token_idx)
                        expert_groups[expert_id][1].append(weight)
        
        # Convert to tensors
        result = {}
        for expert_id, (token_indices, weights) in expert_groups.items():
            if token_indices:
                result[expert_id] = (
                    torch.tensor(token_indices, dtype=torch.long),
                    torch.tensor(weights, dtype=torch.float)
                )
        
        return result
    
    def get_communication_stats(self) -> Dict[str, float]:
        """Get communication statistics."""
        return {
            'cross_gpu_communications': len(set(self.expert_placement.values())),
            'load_imbalance': self._calculate_load_imbalance(),
            'bandwidth_utilization': self._calculate_bandwidth_utilization()
        }
    
    def _calculate_load_imbalance(self) -> float:
        """Calculate load imbalance across GPUs."""
        gpu_loads = [0] * self.num_gpus
        for expert_id, gpu_id in self.expert_placement.items():
            gpu_loads[gpu_id] += 1
        
        if not gpu_loads:
            return 0.0
        
        mean_load = sum(gpu_loads) / len(gpu_loads)
        variance = sum((load - mean_load) ** 2 for load in gpu_loads) / len(gpu_loads)
        return math.sqrt(variance) / (mean_load + 1e-8)
    
    def _calculate_bandwidth_utilization(self) -> float:
        """Calculate average bandwidth utilization."""
        total_bandwidth = 0.0
        count = 0
        
        for i in range(self.num_gpus):
            for j in range(self.num_gpus):
                if i != j:
                    total_bandwidth += self.topology_optimizer.bandwidth_matrix[i, j]
                    count += 1
        
        return total_bandwidth / (count + 1e-8)


class OptimizedMoELayer(nn.Module):
    """
    Highly optimized MoE layer with grouped operations and efficient routing.
    Now supports LaCTMoEExpert.
    """
    def __init__(self, config: MoEConfig, kernel_cost_model: KernelCostModel, 
                 gpu_system_monitor: GpuSystemMonitor):
        super().__init__()
        self.config = config
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        
        self.router = CapacityBasedRouter(config)
        
        self.experts = nn.ModuleList([])
        for i in range(config.num_experts):
            if config.expert_type == "swiglu_ffn":
                self.experts.append(SwiGLUExpert(config, i))
            elif config.expert_type == "quantized":
                self.experts.append(OptimizedQuantizedExpert(config, i)) # config.quantization_bits is in MoEConfig
            elif config.expert_type == "lact_expert": # NEW EXPERT TYPE
                self.experts.append(LaCTMoEExpert(config, i))
            else:
                raise ValueError(f"Unknown expert type: {config.expert_type}")
        
        # Import AdaptiveRouter locally to avoid circular dependency
        from routers import AdaptiveRouter 
        self.adaptive_router = AdaptiveRouter(
            config=config, 
            kernel_cost_model=kernel_cost_model,
            gpu_system_monitor=gpu_system_monitor,
            strategy="kernel_aware_ttha" # Default to adaptive strategy
        )
        
        self.expert_timings: Dict[int, float] = {} # Initialize as Dict for clarity
        
    def forward(self, x: torch.Tensor, use_adaptive_routing: bool = True) -> Tuple[torch.Tensor, Dict]:
        batch_size_seq, d_model = x.shape
        device = x.device
        
        # --- Router Decision ---
        # First, get logits from CapacityBasedRouter's gate
        initial_router_logits = self.router.gate(x) 
        
        # Determine which router to use and get final expert assignments and metadata
        expert_indices, expert_probs, routing_metadata = {}, {}, {} # Initialize to avoid UnboundLocalError
        aux_losses_base = {}

        if use_adaptive_routing and self.gpu_system_monitor is not None:
            expert_indices, expert_probs, routing_metadata = self.adaptive_router(
                initial_router_logits, batch_size_seq 
            )
            # Recompute aux losses based on the *final* assignments from adaptive router
            aux_losses_base = self.router._compute_aux_losses(initial_router_logits, F.softmax(initial_router_logits, dim=-1), expert_indices)
        else:
            # Standard routing via CapacityBasedRouter (no hardware-aware adaptation)
            expert_indices, expert_probs, aux_losses_base = self.router(x) 
            routing_metadata = {}
        
        # --- Expert Computation ---
        if self.config.use_grouped_gemm:
            output = self._grouped_expert_forward(x, expert_indices, expert_probs, routing_metadata)
        else:
            output = self._sequential_expert_forward(x, expert_indices, expert_probs, routing_metadata)
        
        # --- Energy-aware loss (from KCM) ---
        energy_loss = self._compute_energy_aware_loss(expert_indices, expert_probs, batch_size_seq)
        aux_losses_base["energy_loss"] = energy_loss
        
        # --- Collect Comprehensive Metrics ---
        metrics = {
            "aux_losses": aux_losses_base, 
            "expert_usage": self._compute_expert_usage(expert_indices),
            "routing_entropy": self._compute_routing_entropy(expert_probs),
            "routing_metadata": routing_metadata, 
            "expert_timings": self.expert_timings.copy(), 
            "top_k_indices": expert_indices.detach().cpu(), 
            "top_k_probs": expert_probs.detach().cpu(), 
        }
        
        return output, metrics
    
    def _grouped_expert_forward(self, x, expert_indices, expert_probs, router_metadata: Dict[str, Any]):
        batch_size_seq, d_model = x.shape
        output = torch.zeros_like(x)
        
        # Filter out dropped tokens and get their original indices and probabilities
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_probs = expert_probs.view(-1)
        original_token_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).repeat(1, self.config.top_k).view(-1)
        
        valid_mask = (flat_expert_indices != -1) & (flat_expert_probs > 0)
        valid_expert_indices = flat_expert_indices[valid_mask]
        valid_expert_probs = flat_expert_probs[valid_mask]
        valid_original_token_indices = original_token_indices[valid_mask]
        
        if valid_expert_indices.numel() == 0:
            return output

        # Create a combined input tensor for valid tokens and a corresponding scatter map
        # This is essentially the "dispatch" step of an MoE
        unique_expert_ids, counts = torch.unique(valid_expert_indices, return_counts=True)
        
        # Prepare inputs for experts
        expert_inputs_combined = [] # List of tensors, one for each expert's batch
        expert_weights_combined = [] # List of weight tensors
        expert_scatter_indices = []  # List of original token indices for scattering back
        
        current_idx_in_valid = 0
        for expert_id_val in unique_expert_ids:
            expert_mask_for_this_id = (valid_expert_indices == expert_id_val)
            
            tokens_for_this_expert = x[valid_original_token_indices[expert_mask_for_this_id]]
            weights_for_this_expert = valid_expert_probs[expert_mask_for_this_id].unsqueeze(-1) # [num_tokens, 1]
            original_indices_for_this_expert = valid_original_token_indices[expert_mask_for_this_id]
            
            expert_inputs_combined.append(tokens_for_this_expert)
            expert_weights_combined.append(weights_for_this_expert)
            expert_scatter_indices.append(original_indices_for_this_expert)
            
        # Execute experts
        expert_outputs_raw = []
        for i, expert_id_val in enumerate(unique_expert_ids):
            expert_input_batch = expert_inputs_combined[i]
            
            start_time = time.perf_counter()
            if isinstance(self.experts[expert_id_val.item()], LaCTMoEExpert):
                expert_output_raw = self.experts[expert_id_val.item()](expert_input_batch, router_metadata=router_metadata)
            else:
                expert_output_raw = self.experts[expert_id_val.item()](expert_input_batch)
            end_time = time.perf_counter()
            
            self.expert_timings[expert_id_val.item()] = (end_time - start_time) * 1000
            expert_outputs_raw.append(expert_output_raw)

        # Combine outputs (scatter-add equivalent)
        # This requires reconstructing the full output correctly
        all_outputs_to_scatter = []
        all_scatter_indices = []

        for i, expert_output_raw in enumerate(expert_outputs_raw):
            expert_id_val = unique_expert_ids[i]
            weighted_output = expert_output_raw * expert_weights_combined[i]
            
            all_outputs_to_scatter.append(weighted_output)
            all_scatter_indices.append(expert_scatter_indices[i])

        # Concatenate all outputs and indices for a single scatter operation
        final_outputs_to_scatter = torch.cat(all_outputs_to_scatter, dim=0)
        final_scatter_indices = torch.cat(all_scatter_indices, dim=0)
        
        # Perform scatter-add
        output.index_add_(0, final_scatter_indices, final_outputs_to_scatter)
        
        return output
    
    def _sequential_expert_forward(self, x, expert_indices, expert_probs, router_metadata: Dict[str, Any]):
        output = torch.zeros_like(x)
        
        for i in range(x.shape[0]):
            token = x[i:i+1]
            token_output = torch.zeros_like(token)
            
            for k_idx in range(self.config.top_k):
                expert_id = expert_indices[i, k_idx].item()
                weight = expert_probs[i, k_idx].item()
                
                if weight > 0 and expert_id != -1: 
                    if isinstance(self.experts[expert_id], LaCTMoEExpert):
                        expert_out = self.experts[expert_id](token, router_metadata=router_metadata)
                    else:
                        expert_out = self.experts[expert_id](token)
                    token_output += weight * expert_out
            
            output[i] = token_output.squeeze(0)
        
        return output
    
    def _compute_energy_aware_loss(self, expert_indices, expert_probs, num_tokens_in_batch):
        total_predicted_energy = 0.0
        
        # Operations for cost calculation depend on the expert type set in config
        expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
        if self.config.expert_type == "quantized":
            expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 
        elif self.config.expert_type == "lact_expert": 
            expert_ops_for_cost.extend(["lact_fw_forward", "lact_fw_update_loss_grad", "lact_fw_optimizer_step"])

        # Get current hardware state for KCM dynamic cost lookup
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        # Ensure 'temperature' and 'memory_utilization_percent' are present, provide defaults if not
        current_temp = gpu_stats.get('temperature', 0.0)
        current_memory_util = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0

        for i in range(expert_indices.shape[0]): # Iterate over each token in the batch
            token_energy_cost = 0.0
            for k_idx in range(self.config.top_k): # Iterate over chosen experts for this token
                expert_id = expert_indices[i, k_idx].item()
                prob = expert_probs[i, k_idx].item() 
                
                if prob > 0 and expert_id != -1: # Only consider valid, non-zero assignments
                    # Cost for this expert instance for a *single token*
                    expert_total_cost_per_token_instance = 0.0
                    for op_name in expert_ops_for_cost:
                        cost = self.kernel_cost_model.get_cost(
                            op_name, 1, # Use 1 as batch size for base per-token cost
                            current_temp=current_temp, memory_pressure=current_memory_util
                        )
                        expert_total_cost_per_token_instance += cost.get("energy_joules", 0.0)
                    
                    token_energy_cost += expert_total_cost_per_token_instance * prob # Weighted by routing prob

            total_predicted_energy += token_energy_cost # Sum up for the entire batch

        return torch.tensor(total_predicted_energy, device=expert_indices.device, dtype=torch.float32)
    
    def _compute_expert_usage(self, expert_indices):
        usage = torch.zeros(self.config.num_experts, device=expert_indices.device)
        valid_indices = expert_indices[expert_indices != -1]
        
        if valid_indices.numel() > 0:
            usage_counts = torch.bincount(valid_indices.flatten(), minlength=self.config.num_experts).float()
            usage = usage_counts / usage_counts.sum()
        
        return usage
    
    def _compute_routing_entropy(self, expert_probs):
        summed_expert_probs = expert_probs.sum(dim=0) 
        total_prob_sum = summed_expert_probs.sum()
        
        if total_prob_sum == 0:
            return torch.tensor(0.0, device=expert_probs.device) 
        
        normalized_expert_probs = summed_expert_probs / total_prob_sum
        filtered_probs = normalized_expert_probs[normalized_expert_probs > 1e-8]
        
        if filtered_probs.numel() == 0:
            return torch.tensor(0.0, device=expert_probs.device) 
        
        entropy_val = -(filtered_probs * torch.log(filtered_probs)).sum()
        max_entropy = math.log(self.config.num_experts) if self.config.num_experts > 1 else 1.0
        return entropy_val / max_entropy


class MoETransformerBlock(nn.Module):
    """
    Complete MoE transformer block with pre-norm architecture and residual connections.
    """
    def __init__(self, config: MoEConfig, kernel_cost_model: KernelCostModel = None,
                 gpu_system_monitor: GpuSystemMonitor = None):
        super().__init__()
        self.config = config
        
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=1e-6)
        
        self.moe_layer = OptimizedMoELayer(config, kernel_cost_model, gpu_system_monitor)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, use_adaptive_routing: bool = True) -> Tuple[torch.Tensor, Dict]:
        normed_x = self.input_layernorm(x)
        moe_output, metrics = self.moe_layer(normed_x, use_adaptive_routing)
        output = x + self.dropout(moe_output)
        return output, metrics


def compute_total_auxiliary_loss(aux_losses: Dict[str, torch.Tensor], 
                                config: MoEConfig) -> torch.Tensor:
    """Compute weighted sum of all auxiliary losses"""
    device = torch.device('cpu')
    if aux_losses:
        for loss_tensor in aux_losses.values():
            if isinstance(loss_tensor, torch.Tensor):
                device = loss_tensor.device
                break

    total_loss = torch.tensor(0.0, device=device)
    
    if "load_balance_loss" in aux_losses and aux_losses["load_balance_loss"] is not None:
        total_loss += config.load_balance_weight * aux_losses["load_balance_loss"]
    
    if "router_z_loss" in aux_losses and aux_losses["router_z_loss"] is not None:
        total_loss += config.router_z_loss_weight * aux_losses["router_z_loss"]
    
    if "energy_loss" in aux_losses and aux_losses["energy_loss"] is not None:
        total_loss += 0.001 * aux_losses["energy_loss"]
    
    return total_loss


def create_moe_model(config: MoEConfig, kernel_cost_model: KernelCostModel = None,
                    gpu_system_monitor: GpuSystemMonitor = None) -> MoETransformerBlock:
    """Factory function to create MoE model with proper initialization"""
    model = MoETransformerBlock(config, kernel_cost_model, gpu_system_monitor)
    
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    model.apply(init_weights) 
    return model


def get_default_moe_config() -> MoEConfig:
    """Get a reasonable default configuration for MoE models"""
    return MoEConfig(
        d_model=4096,
        num_experts=8,
        top_k=2,
        dropout=0.1,
        expert_type="swiglu_ffn",
        use_grouped_gemm=True,
        load_balance_weight=0.01,
        router_z_loss_weight=0.001,
        capacity_factor=1.25,
        lact_chunk_size=2048,
        lact_lr=1e-3,
        lact_fast_weight_dim_ratio=0.25,
        lact_update_frequency_tokens=1000,
        quantization_bits=8, # Default for general config
        batch_size=64 # Default batch_size for config
    )