import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
from torch.profiler import profile, ProfilerActivity

# Import your NECTAR components
from routers import AdaptiveRouter
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
        
        # Use INT8 quantization for better hardware support
        if quantization_bits == 8:
            self.gate_proj = nn.quantized.Linear(config.d_model, self.hidden_dim)
            self.up_proj = nn.quantized.Linear(config.d_model, self.hidden_dim)
            self.down_proj = nn.quantized.Linear(self.hidden_dim, config.d_model)
        else:
            # Fallback to FP16 for unsupported quantization
            self.gate_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias).half()
            self.up_proj = nn.Linear(config.d_model, self.hidden_dim, bias=config.use_bias).half()
            self.down_proj = nn.Linear(self.hidden_dim, config.d_model, bias=config.use_bias).half()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized operations"""
        # Ensure input is in correct dtype
        if self.quantization_bits == 8:
            x = x.to(torch.float32)  # Quantized ops need FP32 input
        else:
            x = x.half()
        
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # SwiGLU activation
        swish_gate = F.silu(gate_output)
        activated = swish_gate * up_output
        
        output = self.down_proj(activated)
        return output.to(x.dtype)  # Return in original dtype


class CapacityBasedRouter(nn.Module):
    """
    Improved router with capacity-based routing to prevent expert overload.
    Includes auxiliary losses for load balancing and router z-loss.
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.expert_capacity = None  # Will be set dynamically
        
        # Initialize gate weights to small values for stability
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch_size * seq_len, d_model] input tokens
        
        Returns:
            expert_indices: [num_tokens, top_k] selected expert indices
            router_weights: [num_tokens, top_k] routing weights
            aux_losses: Dictionary containing auxiliary losses
        """
        batch_size_seq, d_model = x.shape
        
        # Compute router logits
        router_logits = self.gate(x)  # [num_tokens, num_experts]
        
        # Apply temperature scaling for better routing
        temperature = 1.0
        router_logits = router_logits / temperature
        
        # Compute routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.config.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Normalize probabilities (important for training stability)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Capacity-based routing
        if self.expert_capacity is None:
            # Dynamic capacity based on batch size
            tokens_per_expert = batch_size_seq / self.config.num_experts
            self.expert_capacity = int(tokens_per_expert * self.config.capacity_factor)
        
        # Apply capacity constraints
        top_k_indices, top_k_probs = self._apply_capacity_constraints(
            top_k_indices, top_k_probs, router_probs
        )
        
        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(router_logits, router_probs, top_k_indices)
        
        return top_k_indices, top_k_probs, aux_losses
    
    def _apply_capacity_constraints(self, expert_indices, expert_probs, router_probs):
        """Apply expert capacity constraints to prevent overloading"""
        num_tokens = expert_indices.shape[0]
        
        # Count tokens assigned to each expert
        expert_counts = torch.zeros(self.config.num_experts, device=expert_indices.device)
        for i in range(self.config.num_experts):
            expert_counts[i] = (expert_indices == i).sum()
        
        # If any expert is over capacity, reassign tokens
        over_capacity = expert_counts > self.expert_capacity
        if over_capacity.any():
            # Simple reassignment strategy: drop lowest probability assignments
            mask = torch.ones_like(expert_indices, dtype=torch.bool)
            for expert_id in torch.where(over_capacity)[0]:
                expert_tokens = (expert_indices == expert_id).nonzero()
                if len(expert_tokens) > self.expert_capacity:
                    # Keep only the highest probability assignments
                    probs_for_expert = expert_probs[expert_indices == expert_id]
                    _, keep_indices = torch.topk(probs_for_expert.flatten(), self.expert_capacity)
                    expert_token_indices = expert_tokens[keep_indices // self.config.top_k, 0]
                    # Set mask to False for dropped assignments
                    dropped_mask = torch.ones(len(expert_tokens), dtype=torch.bool)
                    dropped_mask[keep_indices // self.config.top_k] = False
                    mask[expert_tokens[dropped_mask, 0], keep_indices % self.config.top_k] = False
            
            # Apply mask
            expert_indices = expert_indices * mask.long()
            expert_probs = expert_probs * mask.float()
        
        return expert_indices, expert_probs
    
    def _compute_aux_losses(self, router_logits, router_probs, expert_indices):
        """Compute auxiliary losses for training stability"""
        num_tokens = router_logits.shape[0]
        
        # Load balancing loss (encourage uniform expert usage)
        expert_usage = torch.zeros(self.config.num_experts, device=router_logits.device)
        for i in range(self.config.num_experts):
            expert_usage[i] = (expert_indices == i).float().sum()
        
        # Ideal usage is uniform
        ideal_usage = num_tokens * self.config.top_k / self.config.num_experts
        load_balance_loss = F.mse_loss(expert_usage, 
                                     torch.full_like(expert_usage, ideal_usage))
        
        # Router z-loss (encourage router to be decisive)
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        return {
            "load_balance_loss": load_balance_loss,
            "router_z_loss": router_z_loss,
            "expert_usage": expert_usage,
        }


class OptimizedMoELayer(nn.Module):
    """
    Highly optimized MoE layer with grouped operations and efficient routing.
    """
    def __init__(self, config: MoEConfig, kernel_cost_model: KernelCostModel, 
                 gpu_system_monitor: GpuSystemMonitor):
        super().__init__()
        self.config = config
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        
        # Router
        self.router = CapacityBasedRouter(config)
        
        # Experts
        if config.expert_type == "swiglu_ffn":
            self.experts = nn.ModuleList([
                SwiGLUExpert(config, i) for i in range(config.num_experts)
            ])
        elif config.expert_type == "quantized":
            self.experts = nn.ModuleList([
                OptimizedQuantizedExpert(config, i) for i in range(config.num_experts)
            ])
        else:
            raise ValueError(f"Unknown expert type: {config.expert_type}")
        
        # Adaptive router for NECTAR
        self.adaptive_router = AdaptiveRouter(
            config.num_experts, config.top_k,
            kernel_cost_model, gpu_system_monitor,
            "thermal_aware"  # Use thermal-aware routing
        )
        
        # Metrics tracking
        self.expert_timings = {}
        self.expert_usage_history = []
        
    def forward(self, x: torch.Tensor, use_adaptive_routing: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional adaptive routing.
        
        Args:
            x: Input tensor [batch_size * seq_len, d_model]
            use_adaptive_routing: Whether to use NECTAR's adaptive routing
        """
        batch_size_seq, d_model = x.shape
        device = x.device
        
        if use_adaptive_routing and self.gpu_system_monitor is not None:
            # Use NECTAR's adaptive routing
            router_logits = self.router.gate(x)
            expert_indices, expert_probs, routing_metadata = self.adaptive_router(
                router_logits, batch_size_seq
            )
            aux_losses = {"adaptive_routing": torch.tensor(0.0, device=device)}
        else:
            # Standard routing
            expert_indices, expert_probs, aux_losses = self.router(x)
            routing_metadata = {}
        
        # Efficient expert computation using grouped operations
        if self.config.use_grouped_gemm:
            output = self._grouped_expert_forward(x, expert_indices, expert_probs)
        else:
            output = self._sequential_expert_forward(x, expert_indices, expert_probs)
        
        # Compute energy-aware loss if kernel cost model is available
        if self.kernel_cost_model is not None:
            energy_loss = self._compute_energy_aware_loss(expert_indices, expert_probs, batch_size_seq)
            aux_losses["energy_loss"] = energy_loss
        
        # Collect comprehensive metrics
        metrics = {
            "aux_losses": aux_losses,
            "expert_usage": self._compute_expert_usage(expert_indices),
            "routing_entropy": self._compute_routing_entropy(expert_probs),
            "routing_metadata": routing_metadata,
            "expert_timings": self.expert_timings.copy(),
        }
        
        return output, metrics
    
    def _grouped_expert_forward(self, x, expert_indices, expert_probs):
        """Efficient grouped computation of expert outputs"""
        batch_size_seq, d_model = x.shape
        output = torch.zeros_like(x)
        
        # Group tokens by expert for efficient batched computation
        for expert_id in range(self.config.num_experts):
            # Find all tokens assigned to this expert
            expert_mask = (expert_indices == expert_id)
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Extract tokens and weights for this expert
            expert_tokens = x[token_indices]
            expert_weights = torch.zeros(len(token_indices), device=x.device, dtype=x.dtype)
            
            # Compute weights for each token
            for i, token_idx in enumerate(token_indices):
                positions = (expert_indices[token_idx] == expert_id).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    expert_weights[i] = expert_probs[token_idx, positions].sum()
            
            # Time the expert computation
            start_time = time.perf_counter()
            expert_output = self.experts[expert_id](expert_tokens)
            end_time = time.perf_counter()
            
            # Track timing
            self.expert_timings[expert_id] = (end_time - start_time) * 1000  # ms
            
            # Apply weights and accumulate output
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            output[token_indices] += weighted_output
        
        return output
    
    def _sequential_expert_forward(self, x, expert_indices, expert_probs):
        """Sequential expert computation (fallback)"""
        output = torch.zeros_like(x)
        
        for i in range(x.shape[0]):
            token = x[i:i+1]
            token_output = torch.zeros_like(token)
            
            for k in range(self.config.top_k):
                expert_id = expert_indices[i, k].item()
                weight = expert_probs[i, k].item()
                
                if weight > 0:
                    expert_out = self.experts[expert_id](token)
                    token_output += weight * expert_out
            
            output[i] = token_output.squeeze(0)
        
        return output
    
    def _compute_energy_aware_loss(self, expert_indices, expert_probs, batch_size):
        """Compute energy-aware auxiliary loss using kernel cost model"""
        total_energy = 0.0
        
        for expert_id in range(self.config.num_experts):
            expert_mask = (expert_indices == expert_id)
            num_tokens_for_expert = expert_mask.sum().item()
            
            if num_tokens_for_expert > 0:
                # Get energy costs for operations in this expert
                costs = {
                    "ffn_gate": self.kernel_cost_model.get_cost("ffn_gate", num_tokens_for_expert),
                    "ffn_up": self.kernel_cost_model.get_cost("ffn_up", num_tokens_for_expert),
                    "ffn_down": self.kernel_cost_model.get_cost("ffn_down", num_tokens_for_expert),
                    "silu_gelu": self.kernel_cost_model.get_cost("silu_gelu", num_tokens_for_expert),
                }
                
                expert_energy = sum(cost.get("energy_joules", 0.0) for cost in costs.values())
                total_energy += expert_energy
        
        return torch.tensor(total_energy, device=expert_indices.device, dtype=torch.float32)
    
    def _compute_expert_usage(self, expert_indices):
        """Compute expert usage statistics"""
        usage = torch.zeros(self.config.num_experts, device=expert_indices.device)
        for i in range(self.config.num_experts):
            usage[i] = (expert_indices == i).sum().float()
        return usage / usage.sum() if usage.sum() > 0 else usage
    
    def _compute_routing_entropy(self, expert_probs):
        """Compute entropy of routing decisions (higher = more diverse)"""
        # Average probability distribution across all tokens
        avg_probs = expert_probs.mean(dim=0)
        # Compute entropy
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        return entropy


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
    
    model.apply(init_weights)
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
        capacity_factor=1.25
    )