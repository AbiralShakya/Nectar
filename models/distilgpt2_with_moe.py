from transformers import DistilGPT2Model, DistilGPT2Config
import torch.nn as nn
from models.moe_block import MyMoEBlock
from models.ttt_router import SimpleTTTRouter
from typing import Optional

class DistilGPT2WithMoE(DistilGPT2Model):
    """
    Subclass of DistilGPT2Model with MoE blocks replacing the FFN in each transformer layer.
    Allows router swapping and TTT integration.
    """
    def __init__(self, config: DistilGPT2Config, moe_num_experts: int = 4, moe_top_k: int = 2):
        super().__init__(config)
        d_model = config.dim
        # Replace FFN in each transformer block with MyMoEBlock
        for i, layer in enumerate(self.transformer.layer):
            router = SimpleTTTRouter(d_model, moe_num_experts, moe_top_k)
            layer.ffn = MyMoEBlock(d_model, moe_num_experts, router)

    def set_router(self, new_router: nn.Module, layer_idx: Optional[int] = None):
        """
        Swap the router in all or a specific MoE block.
        Args:
            new_router: The new router to use.
            layer_idx: If specified, only swap in the given layer; else all layers.
        """
        if layer_idx is None:
            for layer in self.transformer.layer:
                if hasattr(layer.ffn, 'set_router'):
                    layer.ffn.set_router(new_router)
        else:
            if hasattr(self.transformer.layer[layer_idx].ffn, 'set_router'):
                self.transformer.layer[layer_idx].ffn.set_router(new_router) 