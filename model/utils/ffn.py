import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=4*d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(in_features=4*d_model, out_features=d_model)

    def forward(self, x: torch.Tensor):
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)

        return x