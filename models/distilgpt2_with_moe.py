from transformers import AutoModelForCausalLM
import torch.nn as nn
from models.moe_block import MyMoEBlock
from models.ttt_router import SimpleTTTRouter
from typing import Optional

class DistilGPT2WithMoE(nn.Module):
    """
    Subclass of DistilGPT2Model with MoE blocks replacing the FFN in each transformer layer.
    Uses a causal-LM wrapper to include an LM head that outputs logits directly,
    and allows router swapping and TTT integration.
    """
    def __init__(self, config, moe_num_experts=4, moe_top_k=2):
        super().__init__()
        # load the base causal language model (includes LM head)
        self.transformer = AutoModelForCausalLM.from_config(config)
        d_model = config.hidden_size
        # Replace FFN in each transformer block with MyMoEBlock
        # Under CausalLM wrapper, base model is in .transformer
        for i, layer in enumerate(self.transformer.transformer.h):
            router = SimpleTTTRouter(d_model, moe_num_experts, moe_top_k)
            layer.ffn = MyMoEBlock(d_model, moe_num_experts, router)

    def set_router(self, new_router: nn.Module, layer_idx: Optional[int] = None):
        """
        Swap the router in all or a specific MoE block.
        Args:
            new_router: The new router to use.
            layer_idx: If specified, only swap in the given layer; else all layers.
        """
        layers = self.transformer.transformer.h
        if layer_idx is None:
            for layer in layers:
                if hasattr(layer.ffn, 'set_router'):
                    layer.ffn.set_router(new_router)
        else:
            layer = layers[layer_idx]
            if hasattr(layer.ffn, 'set_router'):
                layer.ffn.set_router(new_router)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass through the causal LM model.
        Returns a ModelOutput with at least .logits (and .loss if labels provided).
        """
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
