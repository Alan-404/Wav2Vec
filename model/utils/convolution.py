import torch
import torch.nn as nn

from typing import Optional

class ExtractionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.norm = nn.LayerNorm(normalized_shape=out_channels)
        self.activation = nn.GELU()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.activation(x)

        return x