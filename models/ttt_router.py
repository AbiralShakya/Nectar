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
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, ttt_lr: float = 1e-3):
        super().__init__(d_model, num_experts, top_k)
        self.ttt_lr = ttt_lr
        self.register_buffer('hardware_bias', torch.zeros(num_experts))
        self.register_buffer('gradient_bias', torch.zeros(num_experts))
        self.ttt_update_count = 0

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        logits = self.gate(x)  # [N, num_experts]
        # Apply hardware and gradient bias
        logits = logits + self.hardware_bias + self.gradient_bias
        # Optionally, add context-based bias
        if ttt_context is not None:
            if 'hardware_signal' in ttt_context:
                logits = logits + ttt_context['hardware_signal']
            if 'gradient_signal' in ttt_context:
                logits = logits + ttt_context['gradient_signal']
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        router_metadata = {'hardware_bias': self.hardware_bias.cpu().numpy(),
                          'gradient_bias': self.gradient_bias.cpu().numpy(),
                          'ttt_update_count': self.ttt_update_count}
        return top_k_indices, top_k_probs, router_metadata

    def ttt_update(self, feedback: Dict[str, Any]):
        """
        Update router state using feedback dict.
        feedback can include 'hardware_stats' (dict), 'gradient_stats' (tensor), etc.
        """
        # Example: update hardware_bias based on power/temp
        if 'hardware_stats' in feedback:
            stats = feedback['hardware_stats']
            # Example: if power > threshold, bias away from expert 0
            if stats.get('power', 0) > 200:
                self.hardware_bias[0] -= self.ttt_lr
            if stats.get('temp', 0) > 70:
                self.hardware_bias[1] -= self.ttt_lr
        # Example: update gradient_bias based on gradient feedback
        if 'gradient_stats' in feedback:
            grad = feedback['gradient_stats']  # [num_experts]
            self.gradient_bias += self.ttt_lr * grad
        self.ttt_update_count += 1 