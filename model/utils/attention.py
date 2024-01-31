import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_samples = int(d_model/heads)

        self.sqrt_dim = math.sqrt(self.head_samples)

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/self.sqrt_dim

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_context = torch.matmul(attention_weights, v)

        return attention_context
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, n_ctx, _ = q.size()

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        qw = qw.view(batch_size, n_ctx, self.heads, self.head_samples).permute([0, 2, 1, 3])
        kw = kw.view(batch_size, n_ctx, self.heads, self.head_samples).permute([0, 2, 1, 3])
        vw = vw.view(batch_size, n_ctx, self.heads, self.head_samples).permute([0, 2, 1, 3])

        attention_output = self.scaled_dot_product_attention(qw, kw, vw, mask)
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, n_ctx, self.d_model)

        attention_output = self.linear_output(attention_output)

        return attention_output