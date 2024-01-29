import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=in_features)
        self.projection = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)

        return x