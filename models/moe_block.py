import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class MyMoEBlock(nn.Module):
    """
    Mixture-of-Experts (MoE) block that can replace a standard FFN in a transformer.
    Supports swappable routers and TTT/hardware feedback integration.
    Input/output shape: [batch, seq, d_model] (matches standard FFN)
    """
    def __init__(self, d_model: int, num_experts: int, router: nn.Module, expert_hidden_dim: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.router = router  # Swappable router (should support TTT/hardware feedback)
        self.expert_hidden_dim = expert_hidden_dim or (4 * d_model)
        # Create experts: each is a small MLP (can be replaced with more complex experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.expert_hidden_dim),
                nn.GELU(),
                nn.Linear(self.expert_hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq, d_model]
            ttt_context: Optional dict with TTT/hardware feedback for the router
        Returns:
            output: [batch, seq, d_model]
            routing_info: dict with routing decisions, diversity, etc.
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, d_model]
        # Router produces expert indices and weights for each token
        expert_indices, expert_weights, router_metadata = self.router(x_flat, ttt_context)
        # expert_indices: [B*S, top_k], expert_weights: [B*S, top_k]
        top_k = expert_indices.shape[1]
        # Dispatch tokens to experts
        expert_outputs = torch.zeros_like(x_flat)
        for k in range(top_k):
            indices = expert_indices[:, k]  # [B*S]
            weights = expert_weights[:, k]  # [B*S]
            for expert_id in range(self.num_experts):
                mask = (indices == expert_id)
                if mask.any():
                    selected = x_flat[mask]
                    out = self.experts[expert_id](selected)
                    expert_outputs[mask] += out * weights[mask].unsqueeze(-1)
        output = expert_outputs.view(B, S, D)
        routing_info = {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'router_metadata': router_metadata
        }
        return output, routing_info

    def set_router(self, new_router: nn.Module):
        """Swap the router (for baseline/energy-aware/TTT, etc)."""
        self.router = new_router 