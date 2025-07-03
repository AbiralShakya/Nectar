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
        # Add energy penalty scaling factor since energy estimates are in micro-joules
        self.energy_scale = 1000.0  # Scale up the penalty
        # Track per-expert energy costs (initialize with uniform distribution)
        self.expert_energy_costs = torch.ones(num_experts) * 1.0  # Normalized costs
        self.expert_usage_count = torch.zeros(num_experts)  # Track usage for adaptive costs

    def _ensure_device_consistency(self, device):
        """Ensure all internal tensors are on the same device."""
        if self.expert_energy_costs.device != device:
            self.expert_energy_costs = self.expert_energy_costs.to(device)
        if self.expert_usage_count.device != device:
            self.expert_usage_count = self.expert_usage_count.to(device)

    def ttt_update(self, feedback: Dict[str, Any]):
        # Store the most recent estimated energy (scalar or per-expert)
        if 'estimated_energy' in feedback:
            self.last_estimated_energy = feedback['estimated_energy']
            
            # Update per-expert energy costs based on usage patterns
            if 'expert_usage' in feedback:
                usage = feedback['expert_usage']  # [num_experts] tensor
                
                # Ensure device consistency (usage should be on the same device as expert_usage_count)
                if usage.device != self.expert_usage_count.device:
                    self.expert_usage_count = self.expert_usage_count.to(usage.device)
                    self.expert_energy_costs = self.expert_energy_costs.to(usage.device)
                
                self.expert_usage_count += usage
                
                # Simple heuristic: more used experts might be more energy-intensive
                # In practice, this would come from actual hardware profiling
                usage_ratio = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
                self.expert_energy_costs = 0.5 + 0.5 * usage_ratio  # Range [0.5, 1.0]
                
        self.ttt_update_count += 1

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Ensure all internal tensors are on the same device as input
        self._ensure_device_consistency(x.device)
        
        logits = self.gate(x)  # [N, num_experts]
        
        # Apply energy penalty with proper scaling and per-expert costs
        if self.last_estimated_energy > 0:
            # Scale the energy penalty to make it meaningful
            base_penalty = self.lambda_energy * self.energy_scale * float(self.last_estimated_energy)
            
            # Apply per-expert energy costs (move to device if needed)
            if x.device != self.expert_energy_costs.device:
                self.expert_energy_costs = self.expert_energy_costs.to(x.device)
            
            # Penalize each expert based on its energy cost
            expert_penalties = base_penalty * self.expert_energy_costs  # [num_experts]
            logits = logits - expert_penalties.unsqueeze(0)  # Broadcast to [N, num_experts]
            
            # Debug: print penalty magnitude
            if self.ttt_update_count % 10 == 0:
                print(f"[Energy Penalty] lambda={self.lambda_energy}, energy={self.last_estimated_energy:.6f}, "
                      f"base_penalty={base_penalty:.6f}, expert_costs={self.expert_energy_costs[:3].tolist()}, "
                      f"logits_range=[{logits.min():.3f}, {logits.max():.3f}]")
        
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Track expert usage for TTT updates
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_usage[i] = (top_k_indices == i).sum().item()
        
        router_metadata = {
            'lambda_energy': self.lambda_energy, 
            'last_estimated_energy': self.last_estimated_energy, 
            'ttt_update_count': self.ttt_update_count,
            'energy_penalty_applied': self.last_estimated_energy > 0,
            'expert_usage': expert_usage,
            'expert_energy_costs': self.expert_energy_costs.cpu().tolist()
        }
        return top_k_indices, top_k_probs, router_metadata 