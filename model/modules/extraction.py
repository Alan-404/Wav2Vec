import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional

from model.utils.convolution import ExtractionLayer

class FeatureExtraction(nn.Module):
    def __init__(self, conv_dims: Union[List[str], Tuple[str]], kernel_sizes: Union[List[str], Tuple[str]], strides: Union[List[str], Tuple[str]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        assert len(kernel_sizes) == len(strides) and len(kernel_sizes) == len(conv_dims)

        self.kernel_sizes = kernel_sizes
        self.strides = strides

        for i in range(len(kernel_sizes)):
            in_channels = conv_dims[i-1] if i != 0 else 1
            self.layers.append(ExtractionLayer(in_channels=in_channels, out_channels=conv_dims[i], kernel_size=kernel_sizes[i], stride=strides[i]))

    def get_lengths(self, lengths: torch.Tensor):
        for idx in range(len(self.kernel_sizes)):
            lengths = (((lengths - self.kernel_sizes[idx]) / self.strides[idx]) + 1).type(torch.int)

        return lengths
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        for layer in self.layers:
            x, lengths = layer(x)

        if lengths is not None:
            lengths = self.get_lengths(lengths)

        return x, lengths