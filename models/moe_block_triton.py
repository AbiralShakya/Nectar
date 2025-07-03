import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

# Import Triton kernels (will be available when Triton is installed)
try:
    from src.triton.moe_dispatch import (
        moe_dispatch_triton, 
        moe_expert_fuse_triton, 
        moe_combine_triton
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, falling back to PyTorch implementation")

class ExpertFFN(nn.Module):
    """Individual expert FFN with up and down projections."""
    def __init__(self, d_model: int, d_ff: int, activation: str = "silu"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        
        # Up projection
        self.w_up = nn.Linear(d_model, d_ff, bias=True)
        # Down projection  
        self.w_down = nn.Linear(d_ff, d_model, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert FFN."""
        # Up projection + activation
        if self.activation == "silu":
            h = F.silu(self.w_up(x))
        elif self.activation == "gelu":
            h = F.gelu(self.w_up(x))
        else:
            h = F.relu(self.w_up(x))
        
        # Down projection
        return self.w_down(h)

class OptimizedMoEBlock(nn.Module):
    """
    Optimized MoE block using Triton kernels for dispatch, expert computation, and combine.
    Falls back to PyTorch implementation if Triton is not available.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 d_ff: Optional[int] = None, router: Optional[nn.Module] = None,
                 use_triton: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff or (4 * d_model)  # Default to 4x expansion
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Router for expert selection
        self.router = router
        
        # Expert FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, self.d_ff) for _ in range(num_experts)
        ])
        
        # Pre-allocate expert buffers for Triton kernels
        if self.use_triton:
            self.register_buffer('expert_buffers', None)
            self.register_buffer('expert_outputs', None)
    
    def set_router(self, router: nn.Module):
        """Set the router for this MoE block."""
        self.router = router
    
    def _allocate_buffers(self, batch_size: int, device: torch.device):
        """Allocate expert buffers for Triton kernels."""
        if not self.use_triton:
            return
            
        # Allocate buffers for each expert
        tokens_per_expert = batch_size // self.num_experts
        if tokens_per_expert == 0:
            tokens_per_expert = 1
            
        self.expert_buffers = [
            torch.empty((tokens_per_expert, self.d_model), 
                       device=device, dtype=torch.float16)
            for _ in range(self.num_experts)
        ]
        
        # Allocate output buffer
        self.expert_outputs = torch.empty(
            (self.num_experts, tokens_per_expert, self.d_model),
            device=device, dtype=torch.float16
        )
    
    def _triton_forward(self, x: torch.Tensor, router_output: Tuple[torch.Tensor, torch.Tensor, Dict]) -> torch.Tensor:
        """Forward pass using Triton kernels."""
        expert_indices, routing_weights, _ = router_output
        N, D = x.shape
        
        # Allocate buffers if needed
        if self.expert_buffers is None:
            self._allocate_buffers(N, x.device)
        
        # Step 1: Dispatch tokens to expert buffers
        moe_dispatch_triton(x, expert_indices, self.expert_buffers, self.top_k)
        
        # Step 2: Fused expert computation
        # Collect expert weights and biases explicitly
        weights_up_list = []
        bias_up_list = []
        weights_down_list = []
        bias_down_list = []
        
        for expert in self.experts:
            weights_up_list.append(expert.w_up.weight)
            bias_up_list.append(expert.w_up.bias)
            weights_down_list.append(expert.w_down.weight)
            bias_down_list.append(expert.w_down.bias)
        
        weights_up = torch.stack(weights_up_list)
        bias_up = torch.stack(bias_up_list)
        weights_down = torch.stack(weights_down_list)
        bias_down = torch.stack(bias_down_list)
        
        moe_expert_fuse_triton(
            self.expert_buffers, weights_up, bias_up, 
            weights_down, bias_down, self.num_experts
        )
        
        # Step 3: Combine expert outputs
        output = torch.empty_like(x)
        moe_combine_triton(
            self.expert_outputs, routing_weights, 
            expert_indices, output, self.top_k
        )
        
        return output
    
    def _pytorch_forward(self, x: torch.Tensor, router_output: Tuple[torch.Tensor, torch.Tensor, Dict]) -> torch.Tensor:
        """Fallback PyTorch implementation."""
        expert_indices, routing_weights, _ = router_output
        N, D = x.shape
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token
        for i in range(N):
            for j in range(self.top_k):
                expert_idx = int(expert_indices[i, j].item())
                weight = routing_weights[i, j].item()
                
                # Apply expert
                expert_output = self.experts[expert_idx](x[i:i+1])
                output[i] += weight * expert_output.squeeze(0)
        
        return output
    
    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Forward pass through the MoE block.
        
        Args:
            x: Input tensor [N, D] where N is number of tokens, D is model dimension
            ttt_context: Optional context for test-time training
            
        Returns:
            Output tensor [N, D]
        """
        if self.router is None:
            raise ValueError("Router must be set before forward pass")
        
        # Get routing decisions
        router_output = self.router(x, ttt_context)
        
        # Use Triton kernels if available and enabled
        if self.use_triton:
            return self._triton_forward(x, router_output)
        else:
            return self._pytorch_forward(x, router_output)
    
    def get_expert_weights(self) -> torch.Tensor:
        """Get all expert weights for analysis."""
        weights = []
        for expert in self.experts:
            expert_weights = {
                'w_up': expert.w_up.weight.data.clone(),
                'b_up': expert.w_up.bias.data.clone(),
                'w_down': expert.w_down.weight.data.clone(),
                'b_down': expert.w_down.bias.data.clone()
            }
            weights.append(expert_weights)
        return weights
    
    def get_expert_usage_stats(self, router_output: Tuple[torch.Tensor, torch.Tensor, Dict]) -> Dict[str, Any]:
        """Get statistics about expert usage."""
        expert_indices, routing_weights, _ = router_output
        
        # Count expert usage
        expert_counts = torch.zeros(self.num_experts, device=expert_indices.device)
        for i in range(expert_indices.shape[0]):
            for j in range(expert_indices.shape[1]):
                expert_idx = expert_indices[i, j].item()
                expert_counts[expert_idx] += 1
        
        # Calculate diversity (fraction of experts used)
        diversity = (expert_counts > 0).float().mean().item()
        
        # Calculate load balancing (standard deviation of expert usage)
        mean_usage = expert_counts.float().mean()
        load_balance = expert_counts.float().std().item()
        
        return {
            'expert_counts': expert_counts.cpu().numpy(),
            'diversity': diversity,
            'load_balance': load_balance,
            'mean_usage': mean_usage.item(),
            'max_usage': expert_counts.max().item(),
            'min_usage': expert_counts.min().item()
        } 