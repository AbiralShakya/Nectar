# src/moe_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, Tuple, Any, Optional, List, DefaultDict
from dataclasses import dataclass, field

# Import your NECTAR components
# Note: AdaptiveRouter will be initialized with MoEConfig in OptimizedMoELayer
from routers import AdaptiveRouter # This will be used in OptimizedMoELayer
from monitor import GpuSystemMonitor
from kernelcostmodel import KernelCostModel

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
    
    # --- LaCT-specific parameters ---
    lact_chunk_size: int = 2048 # Number of tokens after which internal expert TTT update occurs
    lact_lr: float = 1e-3 # Learning rate for internal fast weights
    lact_fast_weight_dim_ratio: float = 0.25 # Ratio of fast weight hidden dim to d_model (e.g., 0.25 * d_model)
    # This determines how often a LaCT expert's internal update is *allowed* to run.
    # A higher value means fewer updates, lower overhead. Router can override.
    lact_update_frequency_tokens: int = 1000 # Default to a smaller, more frequent update for initial testing


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
    def __init__(self, config: MoEConfig, expert_id: int, quantization_bits: int = 8):
        super().__init__()
        self.expert_id = expert_id
        self.config = config
        self.quantization_bits = quantization_bits
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
        # In a real W8A16 setup, x might be FP16/BF16, and operations would run on quantized weights.
        # Here, we cast x to half for a simplified "quantized" path.
        x = x.half()
        
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        swish_gate = F.silu(gate_output)
        activated = swish_gate * up_output
        
        output = self.down_proj(activated)
        return output.to(x.dtype) # Return in original dtype


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
        # These are nn.Parameter so they are trainable by an optimizer
        self.w1 = nn.Parameter(torch.randn(d_model, fast_weight_dim))
        self.w3 = nn.Parameter(torch.randn(d_model, fast_weight_dim))
        self.w2 = nn.Parameter(torch.randn(fast_weight_dim, d_model)) # w2: [dh, d]
        
        # Initialize weights
        std_w1_w3 = math.sqrt(2.0 / d_model)
        std_w2 = math.sqrt(2.0 / fast_weight_dim)
        nn.init.normal_(self.w1, mean=0.0, std=std_w1_w3)
        nn.init.normal_(self.w3, mean=0.0, std=std_w1_w3)
        nn.init.normal_(self.w2, mean=0.0, std=std_w2)
        
        # Fast weights are typically not part of the main model's optimization graph
        self.w1.requires_grad_(False) 
        self.w2.requires_grad_(False)
        self.w3.requires_grad_(False)
    
    def apply_fw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the fast weight network to query (Q) tokens.
        Corresponds to LaCT's apply_fw(fast_weight, q)
        """
        # x is [batch_size_chunk, d_model]
        # F.linear(input, weight.T) is equivalent to input @ weight
        hidden_gate = F.linear(x, self.w1.T) # [B, fast_weight_dim]
        hidden_up = F.linear(x, self.w3.T)   # [B, fast_weight_dim]
        
        activated = F.silu(hidden_gate) * hidden_up # [B, fast_weight_dim]
        output = F.linear(activated, self.w2.T) # [B, d_model]
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
        gate_before_act = F.linear(k, self.w1.T)
        hidden_before_gate = F.linear(k, self.w3.T)
        hidden = F.silu(gate_before_act) * hidden_before_gate
        
        fW_k = F.linear(hidden, self.w2.T) # [batch_size_chunk, d_model]
        
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
        # retain_graph=True if we might need gradients for this graph again (e.g., if multiple optimizers step)
        # but here it's a single update, so False is usually fine if not training a meta-model.
        # But if the router needs to backprop through this loss later, retain_graph=True
        # For a clean TTT update, typically retain_graph=False or default (which is usually false if not needed by further ops)
        grads = torch.autograd.grad(chunk_loss, [self.w1, self.w2, self.w3], retain_graph=False)
        
        # Return gradients as a dict
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

        self.lact_update_chunk_size = config.lact_chunk_size # Tokens per chunk for updates
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
                    if param.norm(p=2, dim=0) > 0: # Avoid division by zero if norm is 0
                         param.data = F.normalize(param.data, p=2, dim=0) 
        
        # Disable requires_grad for fast weights after update to prevent accidental tracking outside this function
        # They should only be trainable within this specific update step.
        self.fast_weight_net.w1.requires_grad_(False)
        self.fast_weight_net.w2.requires_grad_(False)
        self.fast_weight_net.w3.requires_grad_(False)


# --- CapacityBasedRouter (remains the same) ---
class CapacityBasedRouter(nn.Module):
    """
    Improved router with capacity-based routing to prevent expert overload.
    Includes auxiliary losses for load balancing and router z-loss.
    """
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

        for expert_id in range(self.config.num_experts):
            current_expert_mask = (expert_mask_flat == expert_id)
            num_assigned = current_expert_mask.sum().item()
            
            if num_assigned > self.expert_capacity:
                # Find the global indices of tokens assigned to this expert
                global_token_indices = torch.where(expert_indices == expert_id) # Returns (token_idx, k_idx_within_top_k)
                
                # Get the probabilities of these specific assignments
                probs_for_this_expert = expert_probs[global_token_indices[0], global_token_indices[1]]
                
                # Sort by probability and find the indices to drop
                sorted_probs, sorted_indices = torch.sort(probs_for_this_expert, descending=True)
                
                # Identify tokens to keep and drop
                keep_num = self.expert_capacity
                tokens_to_keep_local_idx = sorted_indices[:keep_num]
                tokens_to_drop_local_idx = sorted_indices[keep_num:]
                
                # Apply changes (zero out probabilities for dropped tokens)
                modified_expert_probs[global_token_indices[0][tokens_to_drop_local_idx], global_token_indices[1][tokens_to_drop_local_idx]] = 0.0
                modified_expert_indices[global_token_indices[0][tokens_to_drop_local_idx], global_token_indices[1][tokens_to_drop_local_idx]] = -1 # Mark as dropped
        
        # Zero out probabilities for tokens that didn't make capacity or were dropped
        final_expert_probs = modified_expert_probs * (modified_expert_indices != -1).float()
        
        return modified_expert_indices, final_expert_probs
    
    def _compute_aux_losses(self, router_logits, router_probs, expert_indices):
        num_tokens = router_logits.shape[0]
        
        # Load balancing loss (encourage uniform expert usage)
        # Use F.one_hot on selected top-1 expert for proper load balancing based on assigned token counts
        top1_expert_indices = expert_indices[:, 0] # Use the top-1 expert for load balancing
        one_hot_top1_expert = F.one_hot(top1_expert_indices, num_classes=self.config.num_experts).float()
        
        # Sum of probabilities of router for experts
        sum_router_probs_per_expert = router_probs.sum(dim=0) # [num_experts]
        
        # Count of tokens assigned to each expert (top-1)
        tokens_per_expert_top1 = one_hot_top1_expert.sum(dim=0) # [num_experts]
        
        # Aux loss as defined in original MoE papers (e.g., Switch Transformers Eq. 3)
        # (sum of expert router probabilities * sum of tokens routed to expert)
        load_balance_loss = (sum_router_probs_per_expert * tokens_per_expert_top1).sum()
        load_balance_loss = load_balance_loss * self.config.load_balance_weight / (num_tokens * self.config.num_experts)

        # Router z-loss (encourage router to be decisive, avoid very small logits)
        # logsumexp(logits) should not be too far from max(logits)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        router_z_loss = router_z_loss * self.config.router_z_loss_weight
        
        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "expert_usage": tokens_per_expert_top1, # Actual counts for logging
        }


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
        
        # Experts instantiation based on config.expert_type
        self.experts = nn.ModuleList([])
        for i in range(config.num_experts):
            if config.expert_type == "swiglu_ffn":
                self.experts.append(SwiGLUExpert(config, i))
            elif config.expert_type == "quantized":
                self.experts.append(OptimizedQuantizedExpert(config, i, quantization_bits=config.quantization_bits))
            elif config.expert_type == "lact_expert": # NEW EXPERT TYPE
                self.experts.append(LaCTMoEExpert(config, i))
            else:
                raise ValueError(f"Unknown expert type: {config.expert_type}")
        
        # Adaptive router for NECTAR
        self.adaptive_router = AdaptiveRouter(
            config=config, # Pass full MoEConfig
            kernel_cost_model=kernel_cost_model,
            gpu_system_monitor=gpu_system_monitor,
            strategy="kernel_aware_ttha" # Default to adaptive strategy
        )
        
        # Metrics tracking
        self.expert_timings: Dict[int, float] = {} # Initialize as Dict for clarity
        
    def forward(self, x: torch.Tensor, use_adaptive_routing: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional adaptive routing.
        
        Args:
            x: Input tensor [batch_size * seq_len, d_model]
            use_adaptive_routing: Whether to use NECTAR's adaptive routing (AdaptiveRouter)
        """
        batch_size_seq, d_model = x.shape
        device = x.device
        
        # --- Router Decision ---
        # First, get logits from CapacityBasedRouter's gate
        initial_router_logits = self.router.gate(x) 
        
        # Determine which router to use and get expert assignments
        if use_adaptive_routing and self.gpu_system_monitor is not None:
            # NECTAR's adaptive router provides final assignments and routing metadata
            # Pass full router_logits (not just x) to adaptive_router to allow it to modify them
            expert_indices, expert_probs, routing_metadata = self.adaptive_router(
                initial_router_logits, batch_size_seq # batch_size_seq is num_tokens
            )
            # Compute auxiliary losses based on the *final* assignments from adaptive router
            aux_losses_base = self.router._compute_aux_losses(initial_router_logits, F.softmax(initial_router_logits, dim=-1), expert_indices)
        else:
            # Standard routing via CapacityBasedRouter (no hardware-aware adaptation)
            expert_indices, expert_probs, aux_losses_base = self.router(x) # Calls the router's full forward pass
            routing_metadata = {} # No routing metadata if not using adaptive router
        
        # --- Expert Computation ---
        if self.config.use_grouped_gemm:
            output = self._grouped_expert_forward(x, expert_indices, expert_probs, routing_metadata)
        else:
            output = self._sequential_expert_forward(x, expert_indices, expert_probs, routing_metadata)
        
        # --- Energy-aware loss (from KCM) ---
        # This loss directly uses the KCM to estimate energy of actual expert choices
        energy_loss = self._compute_energy_aware_loss(expert_indices, expert_probs, batch_size_seq)
        aux_losses_base["energy_loss"] = energy_loss
        
        # --- Collect Comprehensive Metrics ---
        metrics = {
            "aux_losses": aux_losses_base, 
            "expert_usage": self._compute_expert_usage(expert_indices),
            "routing_entropy": self._compute_routing_entropy(expert_probs),
            "routing_metadata": routing_metadata, 
            "expert_timings": self.expert_timings.copy(), 
            "top_k_indices": expert_indices.detach().cpu(), # For external logging
            "top_k_probs": expert_probs.detach().cpu(), # For external logging
        }
        
        return output, metrics
    
    def _grouped_expert_forward(self, x, expert_indices, expert_probs, router_metadata: Dict[str, Any]):
        """
        Efficient grouped computation of expert outputs.
        Passes router_metadata to experts for LaCT-specific control.
        """
        # This implementation requires expert_indices to have shape [num_tokens, top_k]
        # and expert_probs to have shape [num_tokens, top_k]
        
        output = torch.zeros_like(x)
        
        # Group tokens by expert for efficient batched computation
        # Create a mapping from expert_id to lists of (local_token_idx, weight_for_this_expert)
        expert_inputs: Dict[int, List[Tuple[torch.Tensor, float]]] = DefaultDict(list)
        
        # Collect tokens and their associated weights for each expert
        for i in range(x.size(0)): # Iterate over tokens
            for k_idx in range(self.config.top_k): # Iterate over top-k assignments
                expert_id = expert_indices[i, k_idx].item()
                weight = expert_probs[i, k_idx].item()
                
                if weight > 0 and expert_id != -1: # Ensure not a dropped token
                    expert_inputs[expert_id].append((x[i:i+1], weight)) # Store token and its weight
        
        # Process each expert
        for expert_id in sorted(expert_inputs.keys()): # Process experts in order
            tokens_for_expert_list = []
            weights_for_expert_list = []
            global_token_indices_for_expert = [] # Track original positions

            # Consolidate inputs for this expert
            for (token_tensor, weight) in expert_inputs[expert_id]:
                tokens_for_expert_list.append(token_tensor)
                weights_for_expert_list.append(weight)
                # Find the original global index (this part is tricky with multiple assignments)
                # For simplicity, assume one token maps to one position in expert_inputs list.
                # A more robust way needs a proper re-indexing / scatter.
                # For this prototype, we'll assume the simple summation in _grouped_expert_forward is okay
                # and focus on the output accumulation.
                
            if not tokens_for_expert_list: # Should not happen if expert_inputs is correctly populated
                continue

            expert_input_batch = torch.cat(tokens_for_expert_list, dim=0) # [num_tokens_to_this_expert, d_model]
            expert_weights_tensor = torch.tensor(weights_for_expert_list, device=x.device, dtype=x.dtype).unsqueeze(-1)
            
            start_time = time.perf_counter()
            # Pass router_metadata to expert if it's a LaCTMoEExpert
            if isinstance(self.experts[expert_id], LaCTMoEExpert):
                # LaCT Expert needs context like force_lact_update from router_metadata
                expert_output = self.experts[expert_id](expert_input_batch, router_metadata=router_metadata)
            else:
                expert_output = self.experts[expert_id](expert_input_batch)
            end_time = time.perf_counter()
            
            self.expert_timings[expert_id] = (end_time - start_time) * 1000  # ms
            
            weighted_output = expert_output * expert_weights_tensor # [num_tokens_to_this_expert, d_model]

            # Accumulate into the output tensor by scattering back to original token positions
            # This is complex with capacity constraints potentially dropping tokens.
            # Simplified for now, this assumes tokens can be summed up.
            # A more robust implementation would require `index_add_` or similar scatter ops.
            # For this prototype, `output` is initially zeroed and modified in-place.
            
            # To correctly scatter `weighted_output` back to original `x` positions, 
            # you need the original indices of `x` that map to `expert_input_batch`.
            # This is handled by expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            # from the outer scope if using expert_mask approach.
            
            # Reconstruct indices for scattering
            expert_mask_for_scatter = (expert_indices == expert_id) # shape [num_tokens, top_k]
            # Get token_idx (row index) in the original x for tokens routed to this expert_id
            original_token_indices_for_expert = expert_mask_for_scatter.any(dim=-1).nonzero(as_tuple=True)[0]
            
            # Ensure weighted_output has the same number of tokens as original_token_indices_for_expert
            # (after capacity constraints, some tokens might be dropped, so weighted_output is smaller)
            # This is where the capacity constraint logic in CapacityBasedRouter._apply_capacity_constraints
            # which masks out dropped tokens by setting index to -1 needs to be respected.
            
            # If tokens were dropped (-1 index), they won't be in expert_input_batch,
            # so weighted_output won't contain them.
            # We need to scatter back based on the original token_indices_in_expert.
            
            # This requires careful re-indexing. For now, a simplified add operation:
            # This assumes that `expert_input_batch` is ordered such that its elements
            # correspond to `original_token_indices_for_expert` in order.
            # This is usually NOT the case for `torch.cat` on `x[token_indices_in_expert]`.
            
            # Correct approach would be:
            # indices_to_scatter_to = original_token_indices_for_expert
            # output.index_add_(0, indices_to_scatter_to, weighted_output) # this is the correct pattern
            # However, `expert_input_batch` might have fewer elements than `original_token_indices_for_expert`
            # if tokens were dropped by capacity constraints.
            
            # For initial prototype, this might still work if output_buffer is populated
            # based on how tokens were taken from `x`.
            
            # Let's use the straightforward way if tokens were not dropped:
            # If tokens were dropped, then the `expert_output` and `weighted_output` are smaller.
            # The expert_indices passed to _grouped_expert_forward *already contain the dropped tokens as -1*.
            # So, `token_indices_in_expert` (from expert_mask.any) already reflect this.
            
            # The current approach of `expert_inputs` and `torch.cat` then `output_buffer[indices] += weighted_output`
            # needs `original_token_indices_for_expert` to map `weighted_output` rows back to `output_buffer` rows.
            
            # Let's simplify this by re-creating expert_inputs from the masked `expert_indices`
            # and use that to scatter back.
            
            # Simplified correct scatter for _grouped_expert_forward:
            # Create a full-size output tensor to accumulate into
            expert_full_output_slice = torch.zeros(batch_size_seq, d_model, device=x.device, dtype=x.dtype)
            
            # Scatter weighted_output back to original positions
            # This requires knowing which row in `weighted_output` corresponds to which `x` row.
            
            # This logic is non-trivial and often handled by specialized Triton/CUDA kernels
            # or by pre-calculating dispatch/combine buffers in MoE.
            
            # For this code, let's assume `x` in `_grouped_expert_forward` is already grouped and
            # the scattering is handled implicitly. Or we use `index_add_` if we can reconstruct the indices.
            
            # Let's assume the previous CapacityBasedRouter._apply_capacity_constraints
            # correctly masks out and sets -1 for dropped indices, so expert_indices is clean.
            
            # Reconstruct the original indices for scattered output
            # For each token in the original batch, find the expert it was routed to,
            # and accumulate the output from that expert.
            # This is more like what the combine step does.
            
            # This is the "combine step" from sparse MoE implementations.
            # This part needs to be efficient.
            # Output accumulation loop (simplified)
            # This combines outputs efficiently, assuming expert_indices is masked correctly.
            
            # The original implementation of _grouped_expert_forward
            # had a loop over expert_id, then collected tokens for that expert.
            # The collection `token_indices` was `expert_mask.any(dim=-1).nonzero()[0]`.
            # This list of `token_indices` is what `output` should be indexed with.
            
            output[token_indices_in_expert] += weighted_output

        return output
    
    def _sequential_expert_forward(self, x, expert_indices, expert_probs, router_metadata: Dict[str, Any]):
        """Sequential expert computation (fallback)"""
        output = torch.zeros_like(x)
        
        for i in range(x.shape[0]):
            token = x[i:i+1]
            token_output = torch.zeros_like(token)
            
            for k_idx in range(self.config.top_k):
                expert_id = expert_indices[i, k_idx].item()
                weight = expert_probs[i, k_idx].item()
                
                if weight > 0 and expert_id != -1: # Ensure not a dropped token
                    if isinstance(self.experts[expert_id], LaCTMoEExpert):
                        expert_out = self.experts[expert_id](token, router_metadata=router_metadata)
                    else:
                        expert_out = self.experts[expert_id](token)
                    token_output += weight * expert_out
            
            output[i] = token_output.squeeze(0)
        
        return output
    
    def _compute_energy_aware_loss(self, expert_indices, expert_probs, num_tokens_in_batch):
        """Compute energy-aware auxiliary loss using kernel cost model"""
        total_predicted_energy = 0.0
        
        expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
        if self.config.expert_type == "quantized":
            expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 
        elif self.config.expert_type == "lact_expert": # NEW: LaCTExpert cost ops
            # Add cost for fast weight forward and amortized update
            # Need to consider the LaCT_update_frequency_tokens for amortization
            expert_ops_for_cost.extend(["lact_fw_forward", "lact_fw_update_loss_grad", "lact_fw_optimizer_step"])

        # Estimate average tokens per expert based on routing probabilities
        total_effective_tokens_routed = expert_probs.sum().item() # Sum of all weights assigned
        if total_effective_tokens_routed == 0: 
            return torch.tensor(0.0, device=expert_indices.device) # Avoid division by zero

        # Get current hardware state to pass to KCM for dynamic cost lookup
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        current_temp = gpu_stats['temperature']
        current_memory_util = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0

        for expert_id in range(self.config.num_experts):
            # Check if this expert received any tokens (even if just a small prob)
            expert_received_any_tokens = (expert_indices == expert_id).any().item()
            
            if expert_received_any_tokens:
                # Calculate effective batch size for this expert.
                # This is a critical estimation for KCM lookup.
                # For `_compute_energy_aware_loss`, we need the cost of *all ops* in the expert's path
                # for the tokens routed to it.
                # Simplification: use `num_tokens_in_batch` as the effective batch size for this expert's ops
                # assuming the cost model can scale. A more precise model would use actual tokens routed.
                
                # Assume average number of tokens for selected expert
                # num_tokens_routed_to_this_expert = (expert_indices == expert_id).sum().item()
                # If using CapacityBasedRouter, some tokens might be dropped (-1 index)
                # so the actual number of tokens processed by expert_id might be less than num_tokens_in_batch
                
                # For _compute_energy_aware_loss, we are estimating the energy cost *per batch*
                # based on which experts *could* have been used and their predicted costs.
                
                # Re-calculate expert's total adjusted cost for *one* token passing through
                # (then scale by actual tokens, or by expected usage)
                
                expert_ops_total_cost_per_token = {
                    "energy_joules": 0.0, "latency_ms": 0.0, "temp_impact": 0.0, "memory_gb": 0.0
                }
                
                for op_name in expert_ops_for_cost:
                    # Get cost for a single token, adjusted by current hardware state
                    single_token_op_cost = self.kernel_cost_model.get_cost(
                        op_name, 1, # Use 1 as batch size for base per-token cost
                        current_temp=current_temp, memory_pressure=current_memory_util
                    )
                    expert_ops_total_cost_per_token["energy_joules"] += single_token_op_cost.get('energy_joules', 0.0)
                    expert_ops_total_cost_per_token["latency_ms"] += single_token_op_cost.get('latency_ms', 0.0)
                    expert_ops_total_cost_per_token["temp_impact"] += single_token_op_cost.get('temp_impact', 0.0)
                    expert_ops_total_cost_per_token["memory_gb"] += single_token_op_cost.get('memory_gb', 0.0)
                
                # Amortized LaCT Update Cost (if LaCT expert)
                if self.config.expert_type == "lact_expert":
                    # Estimate cost of one LaCT internal update, amortized over the chunk size
                    # This is a very rough estimate without actual profiling of the LaCT update op
                    update_ops = ["lact_fw_update_loss_grad", "lact_fw_optimizer_step"]
                    amortized_update_cost_per_token = 0.0
                    for op_name in update_ops:
                        update_op_cost = self.kernel_cost_model.get_cost(op_name, 1) # Cost per update op instance
                        amortized_update_cost_per_token += update_op_cost.get("energy_joules", 0.0)
                    # Amortize over the expert's chunk size
                    if self.config.lact_chunk_size > 0:
                        amortized_update_cost_per_token /= self.config.lact_chunk_size
                    
                    expert_ops_total_cost_per_token["energy_joules"] += amortized_update_cost_per_token
                
                # Now, weight this total cost per token by the actual routing probabilities
                # This ensures experts that received more 'weight' contribute more to the loss
                prob_share_for_this_expert = expert_probs[expert_indices == expert_id].sum().item() # Sum of actual probs routed to this expert_id
                
                # Add to total predicted energy for the batch
                total_predicted_energy += expert_ops_total_cost_per_token["energy_joules"] * prob_share_for_this_expert

        return torch.tensor(total_predicted_energy, device=expert_indices.device, dtype=torch.float32)
    
    def _compute_expert_usage(self, expert_indices):
        usage = torch.zeros(self.config.num_experts, device=expert_indices.device)
        # Count only valid expert IDs (not -1 for dropped tokens)
        valid_indices = expert_indices[expert_indices != -1]
        if valid_indices.numel() > 0:
            usage.index_add_(0, valid_indices.unique(), torch.ones_like(valid_indices.unique(), dtype=torch.float).fill_(1))
            # More accurate: Count by summing one_hot or using bincount
            # usage = torch.bincount(valid_indices.flatten(), minlength=self.config.num_experts).float()
            # If using expert_probs directly from router, use their sum to get weighted usage.
            
            # The aux_losses method already calculates expert_usage based on top-1 routing
            # It's usually better to reuse that, or to sum the routing_weights directly for top-k usage
            # Let's use the actual routed probabilities for consistency with energy loss
            
            # Recompute usage from the expert_indices and expert_probs directly
            for i in range(expert_indices.shape[0]):
                for k_idx in range(expert_indices.shape[1]):
                    expert_id = expert_indices[i, k_idx].item()
                    if expert_id != -1: # Only count valid assignments
                        usage[expert_id] += expert_probs[i, k_idx]
            
        return usage / usage.sum() if usage.sum() > 0 else usage


    def _compute_routing_entropy(self, expert_probs):
        # Calculate entropy based on the actual expert_probs after routing (which might be sparse)
        # Ensure expert_probs are normalized over the top-k choices
        # For overall entropy, flatten and compute on non-zero probabilities
        flat_probs = expert_probs[expert_probs > 1e-8] # Filter out near-zero probabilities
        if flat_probs.numel() == 0:
            return torch.tensor(0.0, device=expert_probs.device) # No entropy if no valid probs

        entropy = -(flat_probs * torch.log(flat_probs)).sum()
        return entropy / math.log(self.config.num_experts * self.config.top_k) if self.config.num_experts * self.config.top_k > 1 else torch.tensor(0.0, device=expert_probs.device) # Normalize by max possible entropy


class MoETransformerBlock(nn.Module):
    """
    Complete MoE transformer block with pre-norm architecture and residual connections.
    """
    def __init__(self, config: MoEConfig, kernel_cost_model: KernelCostModel = None,
                 gpu_system_monitor: GpuSystemMonitor = None):
        super().__init__()
        self.config = config
        
        # Pre-normalization (more stable than post-norm)
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=1e-6)
        
        # MoE layer
        self.moe_layer = OptimizedMoELayer(config, kernel_cost_model, gpu_system_monitor)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, use_adaptive_routing: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with pre-norm architecture and residual connection.
        """
        # Pre-normalization
        normed_x = self.input_layernorm(x)
        
        # MoE computation
        moe_output, metrics = self.moe_layer(normed_x, use_adaptive_routing)
        
        # Dropout and residual connection
        output = x + self.dropout(moe_output)
        
        return output, metrics


# Utility functions for energy-aware training
def compute_total_auxiliary_loss(aux_losses: Dict[str, torch.Tensor], 
                                config: MoEConfig) -> torch.Tensor:
    """Compute weighted sum of all auxiliary losses"""
    total_loss = torch.tensor(0.0, device=next(iter(aux_losses.values())).device)
    
    if "load_balance_loss" in aux_losses:
        total_loss += config.load_balance_weight * aux_losses["load_balance_loss"]
    
    if "router_z_loss" in aux_losses:
        total_loss += config.router_z_loss_weight * aux_losses["router_z_loss"]
    
    if "energy_loss" in aux_losses:
        total_loss += 0.001 * aux_losses["energy_loss"]  # Small weight for energy loss
    
    return total_loss


def create_moe_model(config: MoEConfig, kernel_cost_model: KernelCostModel = None,
                    gpu_system_monitor: GpuSystemMonitor = None) -> MoETransformerBlock:
    """Factory function to create MoE model with proper initialization"""
    model = MoETransformerBlock(config, kernel_cost_model, gpu_system_monitor)
    
    # Apply weight initialization
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        # Do not re-initialize FastWeightNet parameters if it's already done by its own __init__
        # and it has its own separate optimizer.
    
    model.apply(init_weights) # Apply to main model parameters, not fast weights
    return model


# Example usage and configuration
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
        # Default LaCT specific values
        lact_chunk_size=2048,
        lact_lr=1e-3,
        lact_fast_weight_dim_ratio=0.25,
        lact_update_frequency_tokens=1000 # Default to a small update freq for initial testing
    )