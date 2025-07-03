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