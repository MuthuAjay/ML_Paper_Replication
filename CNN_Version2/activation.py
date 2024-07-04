import torch
from typing import Optional, Dict, List, Tuple


class Relu:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.X = x
        self.out = torch.clamp(x, min=0)
        return self.out

    def backward(self,
                 dz: torch.Tensor) -> torch.Tensor:
        x = (self.X > 0).float()
        return dz * x

    def parameters(self):
        return []


class Sigmoid:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        sel
        return 1 / (1 + torch.exp(-x))

    def backward(self, dz: torch.Tensor) -> torch.Tensor:
        return self(dz) * (1 - self(dz))


class Softmax:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.max(dim=1, keepdim=True).values
        self.out = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return self.out

    def parameters(self):
        return []
