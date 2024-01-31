import torch
import torch.nn as nn

class PositionalConvEmbedding(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 128, n_groups: int = 16) -> None:
        super().__init__()
        padding = int((kernel_size - 1) / 2) + 1
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding, groups=n_groups)
        self.num_pad_remove = 1 if kernel_size % 2 == 0 else 0
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x[:, :, : - self.num_pad_remove]
        x = self.activation(x)

        return x

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