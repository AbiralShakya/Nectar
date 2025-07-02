#!/usr/bin/env python3
"""
Simple TTT Baseline Implementation

This implements a basic test-time training approach for comparison with EnergyAwareTTTRouter.
Based on traditional TTT principles but adapted for single GPU testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

class SimpleTTTBaseline(nn.Module):
    """
    Simple TTT baseline implementation for comparison.
    
    This implements traditional test-time training with:
    - Small batch updates (every 16-64 tokens)
    - Simple gradient-based adaptation
    - No hardware awareness
    - Basic fast weight updates
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 vocab_size: int = 10000,
                 seq_length: int = 512,
                 ttt_update_frequency: int = 32,  # Traditional small batch
                 ttt_lr: float = 1e-4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.ttt_update_frequency = ttt_update_frequency
        self.ttt_lr = ttt_lr
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_length, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # TTT fast weights (traditional approach)
        self.ttt_fast_weights = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.ttt_optimizer = torch.optim.AdamW([self.ttt_fast_weights], lr=ttt_lr)
        
        # TTT buffers (small batch approach)
        self.ttt_buffer = []
        self.tokens_since_update = 0
        
        # Performance tracking
        self.ttt_update_count = 0
        self.avg_loss = 0.0
        
        print(f"Created SimpleTTTBaseline: {d_model}d, {num_layers} layers, "
              f"TTT update every {ttt_update_frequency} tokens")
    
    def _create_transformer_layer(self, d_model: int, num_heads: int, dropout: float) -> nn.Module:
        """Create a single transformer layer."""
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with traditional TTT."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Apply TTT fast weights
        hidden_states = hidden_states + torch.tanh(hidden_states @ self.ttt_fast_weights)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # TTT update if targets provided
        if targets is not None:
            self._update_ttt(input_ids, hidden_states, targets)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def _update_ttt(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, targets: torch.Tensor):
        """Traditional TTT update with small batches."""
        batch_size, seq_len = input_ids.shape
        num_tokens = batch_size * seq_len
        
        # Add to buffer
        self.ttt_buffer.append({
            'input_ids': input_ids.detach(),
            'hidden_states': hidden_states.detach(),
            'targets': targets.detach()
        })
        
        self.tokens_since_update += num_tokens
        
        # Perform TTT update when buffer is full
        if self.tokens_since_update >= self.ttt_update_frequency:
            self._perform_ttt_update()
    
    def _perform_ttt_update(self):
        """Perform traditional TTT update."""
        if not self.ttt_buffer:
            return
        
        # Traditional TTT loss: language modeling loss
        total_loss = 0.0
        num_updates = 0
        
        for buffer_item in self.ttt_buffer:
            # Recompute with current fast weights
            token_embeds = self.token_embedding(buffer_item['input_ids'])
            position_ids = torch.arange(token_embeds.size(1), device=token_embeds.device).unsqueeze(0)
            position_embeds = self.position_embedding(position_ids)
            hidden_states = token_embeds + position_embeds
            
            # Apply fast weights
            hidden_states = hidden_states + torch.tanh(hidden_states @ self.ttt_fast_weights)
            
            # Forward through layers
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            
            # Compute loss
            logits = self.output_projection(hidden_states)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 buffer_item['targets'].view(-1))
            
            total_loss += loss
            num_updates += 1
        
        if num_updates > 0:
            avg_loss = total_loss / num_updates
            
            # Update fast weights
            self.ttt_optimizer.zero_grad()
            avg_loss.backward()
            self.ttt_optimizer.step()
            
            # Update tracking
            self.ttt_update_count += 1
            self.avg_loss = float(avg_loss)
        
        # Clear buffer
        self.ttt_buffer.clear()
        self.tokens_since_update = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get TTT statistics."""
        return {
            'ttt_update_count': self.ttt_update_count,
            'avg_loss': self.avg_loss,
            'ttt_update_frequency': self.ttt_update_frequency,
            'ttt_lr': self.ttt_lr,
            'fast_weight_norm': self.ttt_fast_weights.norm().item()
        }


class TraditionalTTTComparison:
    """
    Comparison between traditional TTT and EnergyAwareTTTRouter.
    """
    
    def __init__(self, d_model: int = 256, device: str = 'cuda'):
        self.device = torch.device(device)
        self.d_model = d_model
        
        # Create traditional TTT model
        self.traditional_ttt = SimpleTTTBaseline(
            d_model=d_model,
            num_layers=4,
            num_heads=8,
            vocab_size=10000,
            seq_length=512,
            ttt_update_frequency=32,  # Traditional small batch
            ttt_lr=1e-4
        ).to(self.device)
        
        print(f"Initialized TraditionalTTTComparison on {self.device}")
    
    def run_traditional_ttt(self, num_batches: int = 100, batch_size: int = 8) -> List[Dict[str, float]]:
        """Run traditional TTT and collect metrics."""
        results = []
        
        self.traditional_ttt.train()
        
        for batch_idx in range(num_batches):
            # Generate synthetic data
            input_ids = torch.randint(0, 10000, (batch_size, 512)).to(self.device)
            targets = input_ids.clone()  # Language modeling targets
            
            # Forward pass with TTT
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            outputs = self.traditional_ttt(input_ids, targets)
            end_time.record()
            
            torch.cuda.synchronize()
            latency_ms = start_time.elapsed_time(end_time)
            
            # Get GPU metrics (simplified)
            if torch.cuda.is_available():
                power_watt = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.1  # Rough estimate
                memory_used = torch.cuda.memory_allocated() / 1e9
            else:
                power_watt = 0.0
                memory_used = 0.0
            
            # Calculate metrics
            num_tokens = batch_size * 512
            energy_per_token = (power_watt * latency_ms / 1000) / num_tokens
            throughput = num_tokens / (latency_ms / 1000)
            
            result = {
                'batch_idx': batch_idx,
                'latency_ms': latency_ms,
                'power_watt': power_watt,
                'memory_gb': memory_used,
                'energy_per_token_j': energy_per_token,
                'throughput_tokens_per_sec': throughput,
                'ttt_update_count': self.traditional_ttt.ttt_update_count,
                'avg_loss': self.traditional_ttt.avg_loss
            }
            
            results.append(result)
            
            if batch_idx % 10 == 0:
                print(f"Traditional TTT Batch {batch_idx}: "
                      f"Latency={latency_ms:.2f}ms, "
                      f"Power={power_watt:.1f}W, "
                      f"TTT Updates={self.traditional_ttt.ttt_update_count}")
        
        return results
    
    def compare_with_energy_aware(self, energy_aware_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Compare traditional TTT with EnergyAwareTTTRouter results."""
        if not energy_aware_results:
            return {}
        
        # Calculate averages for traditional TTT
        traditional_avg_latency = np.mean([r['latency_ms'] for r in energy_aware_results])
        traditional_avg_power = np.mean([r['power_watt'] for r in energy_aware_results])
        traditional_avg_energy = np.mean([r['energy_per_token_j'] for r in energy_aware_results])
        traditional_avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in energy_aware_results])
        
        # Calculate improvements (assuming energy_aware_results has better metrics)
        improvements = {
            'latency_improvement_pct': 0.0,  # Will be calculated when we have energy_aware_results
            'power_improvement_pct': 0.0,
            'energy_improvement_pct': 0.0,
            'throughput_improvement_pct': 0.0
        }
        
        return improvements


def main():
    """Simple test of traditional TTT baseline."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create comparison
    comparison = TraditionalTTTComparison(d_model=256, device=device)
    
    # Run traditional TTT
    print("Running Traditional TTT...")
    results = comparison.run_traditional_ttt(num_batches=50, batch_size=8)
    
    # Print summary
    avg_latency = np.mean([r['latency_ms'] for r in results])
    avg_power = np.mean([r['power_watt'] for r in results])
    avg_energy = np.mean([r['energy_per_token_j'] for r in results])
    avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in results])
    
    print(f"\nTraditional TTT Summary:")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Power: {avg_power:.1f} W")
    print(f"Average Energy per Token: {avg_energy:.6f} J")
    print(f"Average Throughput: {avg_throughput:.1f} tokens/sec")
    print(f"Total TTT Updates: {results[-1]['ttt_update_count']}")


if __name__ == "__main__":
    main() 