import torch.nn as nn
import torch
import torch.nn.functional as F

class AdaptiveRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int, thermal_signal_generator):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.thermal_signal_generator = thermal_signal_generator

    def forward(self, gate_logits: torch.Tensor):
        priorities = self.thermal_signal_generator.get_expert_priorities()
        bias = torch.tensor([priorities.get(str(i), 0.0) for i in range(self.num_experts)],
                            device=gate_logits.device)
        biased_logits = gate_logits + bias
        topk_vals, topk_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1)
        return topk_indices, routing_weights

