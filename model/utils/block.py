import torch
import torch.nn as nn

from model.utils.attention import MultiHeadAttention
from model.utils.ffn import FeedForward

from typing import Optional

class Block(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout_rate=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dropout_rate=dropout_rate)

        self.norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm_2 = nn.LayerNorm(normalized_shape=d_model)

        self.dropout_1 = nn.Dropout(p=dropout_rate)
        self.dropout_2 = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor,  mask: Optional[torch.Tensor] = None):
        norm_x = self.norm_1(x)
        attention_output = self.attention(norm_x, norm_x, norm_x, mask)
        attention_output = x + self.dropout_1(attention_output)

        norm_attetion = self.norm_2(attention_output)
        ffn_output = self.ffn(norm_attetion)
        ffn_output = attention_output + self.dropout_2(ffn_output)

        return ffn_output