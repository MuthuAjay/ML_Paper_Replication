import torch


class Relu:

    def __call__(self,
                 X: torch.Tensor):
        self.X = X
        self.out = torch.clamp(X, min=0)
        return self.out

    def backward(self,
                 dZ: torch.Tensor):
        mask = (self.X > 0).float()
        return dZ * mask

    def parameters(self):
        return []


class Softmax:

    def __call__(self,
                 X: torch.Tensor,
                 dim: int):
        X = X - torch.max(X, dim=1, keepdims=True).values
        self.out = torch.exp(X) / torch.sum(torch.exp(X), dim=dim, keepdims=True)
        return self.out

    def parameters(self):
        return []