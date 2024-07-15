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
        self.x = x
        return 1 / (1 + torch.exp(-self.x))

    def backward(self, dz: torch.Tensor) -> torch.Tensor:
        return (self(self.x) * (1 - self(self.x))) * dz


class Softmax:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.max(dim=1, keepdim=True).values
        self.out = torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)
        return self.out

    def parameters(self):
        return []

    def backpropagation(self):
        pass
