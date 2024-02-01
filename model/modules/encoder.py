import torch
import torch.nn as nn

from model.utils.augment import Masked
from model.utils.convolution import PositionalConvEmbedding
from model.utils.block import Block

from typing import Optional

class Encoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(normalized_shape=d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.pe = PositionalConvEmbedding(d_model=d_model)

        self.layers = nn.ModuleList([Block(d_model=d_model, heads=heads, dropout_rate=dropout_rate) for _ in range(n_layers)])
        
        self.post_norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.pre_norm(x)
        x = self.dropout(x)
        
        pos_embedding = self.pe(x.transpose(1,2)).transpose(1,2)
        x += pos_embedding

        for layer in self.layers:
            x = layer(x, mask)

        x = self.post_norm(x)

        return x