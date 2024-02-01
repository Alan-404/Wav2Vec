import torch
import torch.nn as nn
from torchaudio.transforms import TimeMasking


class Masked(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masker = TimeMasking(time_mask_param=10, p=0.065)

    def forward(self, x: torch.Tensor):
        x = self.masker(x)
        return x