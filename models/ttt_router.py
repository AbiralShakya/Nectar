import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class SimpleTTTRouter(nn.Module):
    """
    Simple TTT router that can be updated with feedback.
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.ttt_update_count = 0

    def ttt_update(self, feedback: Dict[str, Any]):
        # Simple TTT update - just count updates
        self.ttt_update_count += 1

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        router_metadata = {'ttt_update_count': self.ttt_update_count}
        return top_k_indices, top_k_probs, router_metadata

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
        # Add adaptive penalty strategy
        self.penalty_strategy = 'load_balance'  # 'uniform' or 'load_balance'
        self.min_expert_penalty = 0.1  # Minimum penalty factor
        self.max_expert_penalty = 2.0  # Maximum penalty factor

    def _ensure_device_consistency(self, device):
        """Ensure all internal tensors are on the same device."""
        if self.expert_energy_costs.device != device:
            self.expert_energy_costs = self.expert_energy_costs.to(device)
        if self.expert_usage_count.device != device:
            self.expert_usage_count = self.expert_usage_count.to(device)

    def _compute_adaptive_penalties(self, base_penalty):
        """Compute adaptive penalties based on expert usage patterns."""
        if self.penalty_strategy == 'uniform':
            # Uniform penalty for all experts
            return torch.ones(self.num_experts, device=self.expert_energy_costs.device) * base_penalty
        elif self.penalty_strategy == 'load_balance':
            # Load balancing: penalize overused experts more
            usage_ratio = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
            
            # Normalize usage ratio to [0, 1]
            if usage_ratio.max() > 0:
                normalized_usage = usage_ratio / usage_ratio.max()
            else:
                normalized_usage = torch.zeros_like(usage_ratio)
            
            # Create penalty factors: higher usage = higher penalty
            penalty_factors = self.min_expert_penalty + (self.max_expert_penalty - self.min_expert_penalty) * normalized_usage
            
            return base_penalty * penalty_factors
        else:
            return torch.ones(self.num_experts, device=self.expert_energy_costs.device) * base_penalty

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
                
        self.ttt_update_count += 1

    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Ensure all internal tensors are on the same device as input
        self._ensure_device_consistency(x.device)
        
        logits = self.gate(x)  # [N, num_experts]
        
        # Apply energy penalty with proper scaling and per-expert costs
        if self.last_estimated_energy > 0:
            # Scale the energy penalty to make it meaningful
            base_penalty = self.lambda_energy * self.energy_scale * float(self.last_estimated_energy)
            
            # Apply adaptive penalties based on usage patterns
            expert_penalties = self._compute_adaptive_penalties(base_penalty)  # [num_experts]
            logits = logits - expert_penalties.unsqueeze(0)  # Broadcast to [N, num_experts]
            
            # Debug: print penalty magnitude
            if self.ttt_update_count % 10 == 0:
                usage_ratio = self.expert_usage_count / (self.expert_usage_count.sum() + 1e-8)
                print(f"[Energy Penalty] lambda={self.lambda_energy}, energy={self.last_estimated_energy:.6f}, "
                      f"base_penalty={base_penalty:.6f}, strategy={self.penalty_strategy}, "
                      f"penalty_range=[{expert_penalties.min():.6f}, {expert_penalties.max():.6f}], "
                      f"usage_range=[{usage_ratio.min():.3f}, {usage_ratio.max():.3f}], "
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
            'expert_energy_costs': self.expert_energy_costs.cpu().tolist(),
            'penalty_strategy': self.penalty_strategy
        }
        return top_k_indices, top_k_probs, router_metadata

# NEW: LaCT Implementation Classes
class SwiGLUGate(nn.Module):
    """
    SwiGLU-based gating network for improved representation capacity.
    Based on Zhang et al.'s LaCT approach for better stability at large chunk sizes.
    """
    def __init__(self, d_model: int, num_experts: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 2  # SwiGLU typically uses 2x hidden dim
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # SwiGLU gate: W1(x) * σ(W2(x))
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, num_experts)
        
        # Layer normalization for stability
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer normalization
        x = self.ln(x)
        
        # SwiGLU activation
        swish = self.w1(x) * torch.sigmoid(self.w2(x))
        
        # Final projection to expert logits
        logits = self.w3(swish)
        
        return logits

class LaCTEnergyAwareTTTRouter(nn.Module):
    """
    Large-Chunk TTT (LaCT) Energy-Aware Router based on Zhang et al.
    
    Key features:
    - Large-chunk updates (accumulate over thousands of tokens)
    - SwiGLU-based gating network for better capacity
    - Muon-style updates with weight normalization
    - Energy-aware penalty with adaptive load balancing
    - Single-GPU optimized (no distributed ops)
    """
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 lambda_energy: float = 0.001, chunk_size: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_energy = lambda_energy
        self.chunk_size = chunk_size  # Large-chunk TTT
        
        # SwiGLU-based gating network
        self.gate = SwiGLUGate(d_model, num_experts)
        
        # Energy penalty parameters
        self.energy_scale = 1000.0
        self.last_estimated_energy = 0.0
        self.ttt_update_count = 0
        
        # Chunk accumulation for large-chunk TTT
        self.chunk_token_count = 0
        self.accumulated_expert_usage = torch.zeros(num_experts)
        self.accumulated_energy_cost = 0.0
        
        # Adaptive penalty strategy
        self.penalty_strategy = 'load_balance'
        self.min_expert_penalty = 0.1
        self.max_expert_penalty = 2.0
        
        # Muon-style update parameters (for stability)
        self.muon_momentum = 0.9
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        self.expert_bias_momentum = torch.zeros(num_experts)
        
        # Weight normalization for stability
        self.weight_norm_enabled = True
        if self.weight_norm_enabled:
            self._apply_weight_norm()
    
    def _apply_weight_norm(self):
        """Apply weight normalization to gate parameters for stability."""
        for name, module in self.gate.named_modules():
            if isinstance(module, nn.Linear):
                nn.utils.weight_norm(module, name='weight', dim=0)
    
    def _ensure_device_consistency(self, device):
        """Ensure all internal tensors are on the same device."""
        if self.accumulated_expert_usage.device != device:
            self.accumulated_expert_usage = self.accumulated_expert_usage.to(device)
        if self.expert_bias_momentum.device != device:
            self.expert_bias_momentum = self.expert_bias_momentum.to(device)
    
    def _compute_adaptive_penalties(self, base_penalty):
        """Compute adaptive penalties based on expert usage patterns."""
        if self.penalty_strategy == 'uniform':
            return torch.ones(self.num_experts, device=self.expert_bias.device) * base_penalty
        elif self.penalty_strategy == 'load_balance':
            # Use accumulated usage for more stable penalties
            usage_ratio = self.accumulated_expert_usage / (self.accumulated_expert_usage.sum() + 1e-8)
            
            if usage_ratio.max() > 0:
                normalized_usage = usage_ratio / usage_ratio.max()
            else:
                normalized_usage = torch.zeros_like(usage_ratio)
            
            penalty_factors = self.min_expert_penalty + (self.max_expert_penalty - self.min_expert_penalty) * normalized_usage
            return base_penalty * penalty_factors
        else:
            return torch.ones(self.num_experts, device=self.expert_bias.device) * base_penalty
    
    def _muon_update(self, expert_gradients: torch.Tensor):
        """
        Muon-style update for expert bias parameters.
        Based on Zhang et al.'s approach for stable large-chunk updates.
        """
        # Update momentum
        self.expert_bias_momentum = self.muon_momentum * self.expert_bias_momentum + (1 - self.muon_momentum) * expert_gradients
        
        # Apply update with learning rate scaling
        lr = 0.01  # Could be made configurable
        self.expert_bias.data -= lr * self.expert_bias_momentum
        
        # Weight normalization for stability
        if self.weight_norm_enabled:
            with torch.no_grad():
                self.expert_bias.data = F.normalize(self.expert_bias.data, p=2, dim=0) * torch.sqrt(torch.tensor(self.num_experts))
    
    def ttt_update(self, feedback: Dict[str, Any]):
        """
        Large-chunk TTT update based on Zhang et al.
        Accumulates updates over chunk_size tokens before applying.
        """
        # Accumulate energy cost
        if 'estimated_energy' in feedback:
            self.accumulated_energy_cost += feedback['estimated_energy']
        
        # Accumulate expert usage
        if 'expert_usage' in feedback:
            usage = feedback['expert_usage']
            if usage.device != self.accumulated_expert_usage.device:
                self.accumulated_expert_usage = self.accumulated_expert_usage.to(usage.device)
            self.accumulated_expert_usage += usage
        
        # Count tokens in current chunk
        if 'token_count' in feedback:
            self.chunk_token_count += feedback['token_count']
        else:
            # Estimate token count from batch size and sequence length
            self.chunk_token_count += feedback.get('batch_size', 1) * feedback.get('seq_length', 64)
        
        # Check if we should perform large-chunk update
        if self.chunk_token_count >= self.chunk_size:
            self._perform_chunk_update()
    
    def _perform_chunk_update(self):
        """Perform the actual large-chunk update."""
        if self.chunk_token_count == 0:
            return
        
        # Compute average energy cost per token
        avg_energy = self.accumulated_energy_cost / self.chunk_token_count
        self.last_estimated_energy = avg_energy
        
        # Compute expert gradients based on accumulated usage
        usage_ratio = self.accumulated_expert_usage / (self.accumulated_expert_usage.sum() + 1e-8)
        
        # Muon-style gradient computation
        # Penalize overused experts, reward underused experts
        target_usage = torch.ones_like(usage_ratio) / self.num_experts  # Uniform target
        expert_gradients = (usage_ratio - target_usage) * self.lambda_energy * self.energy_scale
        
        # Apply Muon update
        self._muon_update(expert_gradients)
        
        # Reset accumulation
        self.accumulated_expert_usage.zero_()
        self.accumulated_energy_cost = 0.0
        self.chunk_token_count = 0
        self.ttt_update_count += 1
        
        # Debug output
        if self.ttt_update_count % 5 == 0:
            print(f"[LaCT Update] Chunk {self.ttt_update_count}: avg_energy={avg_energy:.6f}, "
                  f"usage_range=[{usage_ratio.min():.3f}, {usage_ratio.max():.3f}], "
                  f"bias_range=[{self.expert_bias.min():.3f}, {self.expert_bias.max():.3f}]")
    
    def forward(self, x: torch.Tensor, ttt_context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Ensure device consistency
        self._ensure_device_consistency(x.device)
        
        # SwiGLU-based gating
        logits = self.gate(x)  # [N, num_experts]
        
        # Add expert bias (Muon-style)
        logits = logits + self.expert_bias.unsqueeze(0)
        
        # Apply energy penalty
        if self.last_estimated_energy > 0:
            base_penalty = self.lambda_energy * self.energy_scale * float(self.last_estimated_energy)
            expert_penalties = self._compute_adaptive_penalties(base_penalty)
            logits = logits - expert_penalties.unsqueeze(0)
        
        # Softmax and top-k selection
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Track expert usage for chunk accumulation
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_usage[i] = (top_k_indices == i).sum().item()
        
        router_metadata = {
            'lambda_energy': self.lambda_energy,
            'last_estimated_energy': self.last_estimated_energy,
            'ttt_update_count': self.ttt_update_count,
            'chunk_token_count': self.chunk_token_count,
            'energy_penalty_applied': self.last_estimated_energy > 0,
            'expert_usage': expert_usage,
            'penalty_strategy': self.penalty_strategy,
            'router_type': 'LaCT_Energy_Aware'
        }
        
        return top_k_indices, top_k_probs, router_metadata