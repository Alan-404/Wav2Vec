import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, token_size: int, d_model: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(in_features=d_model, out_features=token_size)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.linear(x)

        return x
