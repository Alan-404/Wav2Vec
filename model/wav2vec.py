import torch
import torch.nn as nn

from model.modules.extraction import FeatureExtraction
from model.modules.projection import FeatureProjection
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder
from model.utils.masking import generate_mask

from typing import List, Union, Tuple, Optional

class Wav2Vec(nn.Module):
    def __init__(self,
                 token_size: int,
                 n_layers: int,
                 d_model: int,
                 heads: int,
                 conv_dims: Union[List[str], Tuple[str]] = (512, 512, 512, 512, 512, 512, 512, 512), 
                 kernel_sizes: Union[List[str], Tuple[str]] = (10, 3, 3, 3, 2, 2, 2, 2), 
                 strides: Union[List[str], Tuple[str]] = (5, 2, 2, 2, 2, 2, 2, 2),
                 dropout_rate: float = 0.0) -> None:
        super().__init__()

        self.extraction = FeatureExtraction(conv_dims=conv_dims, kernel_sizes=kernel_sizes, strides=strides)
        self.projection = FeatureProjection(in_features=conv_dims[-1], out_features=d_model, dropout_rate=dropout_rate)

        self.encoder = Encoder(n_layers=n_layers, d_model=d_model, heads=heads, dropout_rate=dropout_rate)

        self.decoder = Decoder(token_size=token_size, d_model=d_model, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x = x.unsqueeze(1)
        x, lengths = self.extraction(x, lengths)
        x = x.transpose(1, 2)

        mask = None
        if lengths is not None:
            mask = generate_mask(lengths).to(x.device)
            mask = (mask == 0).unsqueeze(1).unsqueeze(1)

        x = self.projection(x)
        x = self.encoder(x, mask)

        x = self.decoder(x)

        if lengths is not None:
            return x, lengths
        return x