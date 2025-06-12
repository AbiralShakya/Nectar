import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class TopKRouter(nn.Module):
    """
    Top-K router for MoE with load balancing loss and detailed profiling.
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        gate_noise: float = 1e-2,
        expert_dropout: float = 0.0,
        balance_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_noise = gate_noise
        self.expert_dropout = expert_dropout
        self.balance_loss_weight = balance_loss_weight
        
        # Gating network - simple linear layer
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # Initialize gate weights
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # For tracking expert usage statistics
        self.register_buffer('expert_usage_counts', torch.zeros(n_experts))
        self.register_buffer('total_tokens_processed', torch.tensor(0.0))
        
    def add_noise(self, logits: torch.Tensor) -> torch.Tensor:
        """Add noise to gate logits for better exploration."""
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(logits) * self.gate_noise
            return logits + noise
        return logits
    
    def compute_load_balancing_loss(
        self, 
        gate_logits: torch.Tensor, 
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.
        
        Args:
            gate_logits: Raw gate logits [batch_size * seq_len, n_experts]
            selected_experts: Selected expert indices [batch_size * seq_len, top_k]
            
        Returns:
            Load balancing loss scalar
        """
        # Compute gate probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B*T, n_experts]
        
        # Fraction of tokens assigned to each expert
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts).float()  # [B*T, top_k, n_experts]
        expert_assignment = expert_mask.sum(dim=1)  # [B*T, n_experts]
        tokens_per_expert = expert_assignment.sum(dim=0)  # [n_experts]
        
        # Normalize by total tokens
        total_tokens = gate_logits.shape[0] * self.top_k
        fraction_per_expert = tokens_per_expert / total_tokens
        
        # Average gate probability for each expert
        avg_gate_prob = gate_probs.mean(dim=0)  # [n_experts]
        
        # Load balancing loss: minimize the dot product of these two distributions
        # This encourages both distributions to be uniform
        balance_loss = (fraction_per_expert * avg_gate_prob).sum() * self.n_experts
        
        return balance_loss
    
    def compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """Compute expert capacity based on capacity factor."""
        tokens_per_expert = (batch_size * seq_len * self.top_k) / self.n_experts
        capacity = int(tokens_per_expert * self.capacity_factor)
        return max(capacity, 4)  # Minimum capacity
    
    def profile_expert_timing(self, selected_experts: torch.Tensor) -> Dict:
        """
        Profile per-expert timing and usage.
        
        Args:
            selected_experts: Selected expert indices [batch_size * seq_len, top_k]
            
        Returns:
            Dictionary with profiling information
        """
        metrics = {}
        
        # Count expert usage
        expert_counts = torch.zeros(self.n_experts, device=selected_experts.device)
        for expert_id in range(self.n_experts):
            expert_counts[expert_id] = (selected_experts == expert_id).sum().float()
        
        # Update global statistics
        self.expert_usage_counts += expert_counts
        self.total_tokens_processed += selected_experts.numel()
        
        # Compute usage statistics
        total_assignments = expert_counts.sum()
        expert_utilization = expert_counts / (total_assignments + 1e-8)
        
        metrics.update({
            'expert_usage_current': expert_counts.cpu().numpy(),
            'expert_utilization_current': expert_utilization.cpu().numpy(),
            'expert_usage_cumulative': self.expert_usage_counts.cpu().numpy(),
            'total_assignments': total_assignments.item(),
            'usage_variance': expert_utilization.var().item(),
            'max_expert_usage': expert_utilization.max().item(),
            'min_expert_usage': expert_utilization.min().item(),
        })
        
        return metrics
    
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Forward pass of the router.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary containing routing decisions and metrics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for easier processing
        x_flat = x.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Timing for gate computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gate_start = torch.cuda.Event(enable_timing=True)
            gate_end = torch.cuda.Event(enable_timing=True)
            gate_start.record()
        
        # Compute gate logits
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, n_experts]
        
        # Add noise for exploration
        gate_logits = self.add_noise(gate_logits)
        
        # Get top-k experts
        top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        if torch.cuda.is_available():
            gate_end.record()
            torch.cuda.synchronize()
            gate_time = gate_start.elapsed_time(gate_end)
        else:
            gate_time = 0.0
        
        # Compute routing probabilities (softmax over top-k)
        top_k_probs = F.softmax(top_k_values, dim=-1)  # [batch_size * seq_len, top_k]
        
        # Compute load balancing loss
        balance_loss = self.compute_load_balancing_loss(gate_logits, top_k_indices)
        
        # Profile expert usage
        usage_metrics = self.profile_expert_timing(top_k_indices)
        
        # Compute expert capacity
        capacity = self.compute_capacity(batch_size, seq_len)
        
        # Apply expert dropout during training
        if self.training and self.expert_dropout > 0:
            dropout_mask = torch.rand_like(top_k_probs) > self.expert_dropout
            top_k_probs = top_k_probs * dropout_mask
            # Renormalize
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Prepare output
        output = {
            'expert_indices': top_k_indices,  # [batch_size * seq_len, top_k]
            'expert_weights': top_k_probs,    # [batch_size * seq_len, top_k]
            'gate_logits': gate_logits,       # [batch_size * seq_len, n_experts]
            'balance_loss': balance_loss * self.balance_loss_weight,
            'capacity': capacity,
            'metrics': {
                'gate_computation_time_ms': gate_time,
                'balance_loss_raw': balance_loss.item(),
                'balance_loss_weighted': (balance_loss * self.balance_loss_weight).item(),
                'capacity': capacity,
                'avg_top_k_confidence': top_k_probs.mean().item(),
                'gate_entropy': -F.softmax(gate_logits, dim=-1).mul(F.log_softmax(gate_logits, dim=-1)).sum(-1).mean().item(),
                **usage_metrics,
            }
        }
        
        return output
    
    def get_expert_stats(self) -> Dict:
        """Get comprehensive expert usage statistics."""
        if self.total_tokens_processed == 0:
            return {'message': 'No tokens processed yet'}
        
        cumulative_usage = self.expert_usage_counts / self.total_tokens_processed
        
        return {
            'total_tokens_processed': self.total_tokens_processed.item(),
            'expert_usage_distribution': cumulative_usage.cpu().numpy(),
            'usage_std': cumulative_usage.std().item(),
            'usage_coefficient_of_variation': cumulative_usage.std().item() / (cumulative_usage.mean().item() + 1e-8),
            'most_used_expert': cumulative_usage.argmax().item(),
            'least_used_expert': cumulative_usage.argmin().item(),
            'perfect_balance_target': 1.0 / self.n_experts,
        }
    
    def reset_stats(self):
        """Reset expert usage statistics."""
        self.expert_usage_counts.zero_()
        self.total_tokens_processed.zero_()


# Auxiliary router for comparison/experimentation
class SwitchRouter(TopKRouter):
    """
    Switch Transformer style router (top-1 with capacity dropping).
    Inherits from TopKRouter but modifies behavior for top-1 routing.
    """
    
    def __init__(self, d_model: int, n_experts: int, **kwargs):
        # Force top_k = 1 for Switch routing
        super().__init__(d_model, n_experts, top_k=1, **kwargs)
        
    def forward(self, x: torch.Tensor) -> Dict:
        """Switch-style routing with capacity dropping."""
        # Get base routing decisions
        output = super().forward(x)
        
        # For Switch, we need to implement capacity dropping
        batch_size, seq_len = x.shape[:2]
        capacity = output['capacity']
        expert_indices = output['expert_indices'].squeeze(-1)  # [batch_size * seq_len]
        
        # Count tokens per expert
        expert_counts = torch.zeros(self.n_experts, device=x.device)
        for expert_id in range(self.n_experts):
            expert_counts[expert_id] = (expert_indices == expert_id).sum()
        
        # Create capacity mask
        capacity_mask = torch.ones_like(expert_indices, dtype=torch.bool)
        
        for expert_id in range(self.n_experts):
            expert_tokens = (expert_indices == expert_id).nonzero(as_tuple=True)[0]
            if len(expert_tokens) > capacity:
                # Randomly drop tokens exceeding capacity
                dropped_indices = expert_tokens[capacity:]
                capacity_mask[dropped_indices] = False
        
        # Update metrics
        tokens_dropped = (~capacity_mask).sum().item()
        output['metrics']['tokens_dropped'] = tokens_dropped
        output['metrics']['drop_rate'] = tokens_dropped / expert_indices.numel()
        output['capacity_mask'] = capacity_mask
        
        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the router
    d_model = 256
    n_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 32
    
    router = TopKRouter(
        d_model=d_model,
        n_experts=n_experts,
        top_k=top_k,
        capacity_factor=1.5,
        balance_loss_weight=0.01,
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    with torch.no_grad():
        output = router(x)
    
    print("Router output keys:", output.keys())
    print(f"Expert indices shape: {output['expert_indices'].shape}")
    print(f"Expert weights shape: {output['expert_weights'].shape}")
    print(f"Balance loss: {output['balance_loss'].item():.6f}")
    print(f"Expert capacity: {output['capacity']}")
    
    print("\nMetrics:")
    for key, value in output['metrics'].items():
        if isinstance(value, (list, np.ndarray)):
            print(f"  {key}: {np.array(value)}")
        else:
            print(f"  {key}: {value}")
    
    print("\nExpert statistics:")
    stats = router.get_expert_stats()
    for key, value in stats.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"  {key}: {np.array(value)}")
        else:
            print(f"  {key}: {value}")
    
    # Test Switch router
    print("\n" + "="*50)
    print("Testing Switch Router")
    
    switch_router = SwitchRouter(d_model=d_model, n_experts=n_experts)
    
    with torch.no_grad():
        switch_output = switch_router(x)
    
    print(f"Switch expert indices shape: {switch_output['expert_indices'].shape}")
    print(f"Tokens dropped: {switch_output['metrics']['tokens_dropped']}")
    print(f"Drop rate: {switch_output['metrics']['drop_rate']:.3f}")