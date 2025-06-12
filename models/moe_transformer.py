import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np 
from fairscale.nn.moe import MOELayer
from models.router import TopKRouter

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_experts: int = 8, 
                 top_k: int = 2, dropout: float = 0.1, use_moe: bool = True, capacity_factor: float = 1.25):
        super().__init__()
        self.d_model = d_model
        self.use_moe = use_moe

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.normal = nn.LayerNorm(d_model)

        if use_moe:
            expert = nn.Sequential(nn.Linear(d_model, d_ff), 
                                   nn.ReLU(), 
                                   nn.Dropout(dropout), 
                                   nn.Linear(d_ff, d_model))
            self.router = TopKRouter(
                d_model = d_model, 
                n_experts = n_experts, 
                top_k = top_k, 
                capacity_factor = capacity_factor
            )

            self.moe = MOELayer(self.router.gate, experts=[expert for i in range(n_experts)])

        else:
            self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ff), 
                                              nn.ReLU(),
                                              nn.Dropout(dropout), 
                                              nn.Linear(d_ff, d_model))
            

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.expert_timings = {}

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        profile: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with optional profiling.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            profile: Whether to collect timing information
            
        Returns:
            output: Transformed tensor
            metrics: Dictionary containing routing metrics and timings
        """
        metrics = {}
        
        # Self-attention
        residual = x
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(residual + self.dropout(attn_out))
        
        # MoE or FFN
        residual = x
        
        if self.use_moe:
            # Profile MoE forward pass
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                
            # Get routing decisions first for profiling
            if profile:
                router_output = self.router(x)
                metrics.update(router_output['metrics'])
            
            # MoE forward pass
            moe_out, moe_loss = self.moe(x)
            
            if profile and torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                
                total_time = start_event.elapsed_time(end_event)
                metrics['moe_forward_time_ms'] = total_time
                
                # Per-expert profiling (approximate)
                if hasattr(self.moe, 'gate') and hasattr(self.moe.gate, 'wg'):
                    gate_scores = F.softmax(torch.matmul(x, self.moe.gate.wg.weight.T), dim=-1)
                    expert_usage = gate_scores.mean(dim=[0, 1])  # Average usage per expert
                    metrics['expert_usage'] = expert_usage.cpu().numpy()
            
            x = residual + self.dropout(moe_out)
            metrics['aux_loss'] = moe_loss
            
        else:
            # Standard FFN
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                ffn_out = self.ffn(x)
                end_event.record()
                torch.cuda.synchronize()
                
                metrics['ffn_time_ms'] = start_event.elapsed_time(end_event)
            else:
                ffn_out = self.ffn(x)
                
            x = residual + self.dropout(ffn_out)
        
        x = self.norm2(x)
        
        return x, metrics


class MoETransformer(nn.Module):
    """Simple MoE Transformer model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        n_experts: int = 8,
        top_k: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_moe_layers: Optional[list] = None,  # Which layers use MoE
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Determine which layers use MoE
        if use_moe_layers is None:
            # By default, use MoE in every other layer starting from layer 1
            use_moe_layers = [i % 2 == 1 for i in range(n_layers)]
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_experts=n_experts,
                top_k=top_k,
                dropout=dropout,
                use_moe=use_moe_layers[i],
                capacity_factor=capacity_factor,
            )
            for i in range(n_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        profile: bool = False
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            profile: Whether to collect profiling information
            
        Returns:
            Dictionary containing logits, aux_loss, and optional metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Attention mask for causal modeling
        if attention_mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
        else:
            causal_mask = attention_mask
        
        # Forward through layers
        total_aux_loss = 0.0
        all_metrics = {} if profile else None
        
        for i, layer in enumerate(self.layers):
            x, layer_metrics = layer(x, mask=causal_mask, profile=profile)
            
            # Accumulate auxiliary loss from MoE layers
            if 'aux_loss' in layer_metrics:
                total_aux_loss += layer_metrics['aux_loss']
            
            # Collect metrics
            if profile:
                for key, value in layer_metrics.items():
                    if key != 'aux_loss':
                        all_metrics[f'layer_{i}_{key}'] = value
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        output = {
            'logits': logits,
            'aux_loss': total_aux_loss,
        }
        
        if profile:
            output['metrics'] = all_metrics
            
        return output


# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = MoETransformer(
        vocab_size=1000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        n_experts=4,
        top_k=2,
        use_moe_layers=[False, True, False, True],  # MoE in layers 1 and 3
    )
    
    # Test input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass with profiling
    with torch.no_grad():
        output = model(input_ids, profile=True)
        
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Auxiliary loss: {output['aux_loss'].item():.6f}")
    
    if 'metrics' in output:
        print("\nProfiling metrics:")
        for key, value in output['metrics'].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")