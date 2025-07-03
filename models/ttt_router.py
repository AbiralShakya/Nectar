import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

class SimpleTTTRouter(nn.Module):
    """
    Simple router for MoE that supports TTT and hardware feedback.
    Produces top-k expert indices and weights for each token.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        # Optionally, add TTT state/parameters here

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [N, d_model] (flattened tokens)
            ttt_context: Optional dict with TTT/hardware feedback
        Returns:
            expert_indices: [N, top_k]
            expert_weights: [N, top_k]
            router_metadata: dict with any extra info
        """
        logits = self.gate(x)  # [N, num_experts]
        # Optionally, adjust logits using ttt_context/hardware feedback
        if ttt_context is not None and 'hardware_signal' in ttt_context:
            logits = logits + ttt_context['hardware_signal']  # Example: add bias
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        router_metadata = {}
        return top_k_indices, top_k_probs, router_metadata

    def ttt_update(self, feedback: Dict[str, Any]):
        """Perform a TTT update using feedback (e.g., hardware stats, gradients)."""
        # Implement TTT update logic here (e.g., update internal state, adapt parameters)
        pass 

class EnergyAwareTTTRouter(SimpleTTTRouter):
    """
    Energy-aware TTT router that adapts routing based on hardware and gradient feedback.
    Maintains state for TTT updates and applies feedback to routing logits.
    Now includes an explicit energy penalty (lambda_energy * estimated_energy) in the gating score.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, lambda_energy: float = 0.001):
        super().__init__(d_model, num_experts, top_k)
        self.lambda_energy = lambda_energy
        self.last_estimated_energy = 0.0  # Scalar or tensor (per-expert)
        self.ttt_update_count = 0

    def ttt_update(self, feedback: Dict[str, Any]):
        # Store the most recent estimated energy (scalar or per-expert)
        if 'estimated_energy' in feedback:
            self.last_estimated_energy = feedback['estimated_energy']
        self.ttt_update_count += 1

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        logits = self.gate(x)  # [N, num_experts]
        # Subtract energy penalty (broadcast as needed)
        if isinstance(self.last_estimated_energy, torch.Tensor):
            # Per-expert energy: shape [num_experts]
            penalty = self.lambda_energy * self.last_estimated_energy
            logits = logits - penalty.unsqueeze(0)  # Broadcast to [N, num_experts]
        else:
            # Scalar energy: subtract from all logits
            logits = logits - self.lambda_energy * float(self.last_estimated_energy)
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        router_metadata = {'lambda_energy': self.lambda_energy, 'last_estimated_energy': self.last_estimated_energy, 'ttt_update_count': self.ttt_update_count}
        return top_k_indices, top_k_probs, router_metadata 