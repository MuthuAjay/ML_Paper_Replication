import torch
from torch import nn


class VGG16(nn.Module):

    def __init__(self,
                 input_shape: int,
                 output_shape=int):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return x
