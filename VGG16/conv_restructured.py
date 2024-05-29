from itertools import repeat

import torch
from typing import Optional, List, Tuple


class Relu:

    def __call__(self,
                 X: torch.Tensor):
        return torch.clamp(X, min=0)

    def backward(self,
                 dZ: torch.Tensor):
        return (dZ > 0).float

    def parameters(self):
        return []


class CrossEntropyLoss:

    def __call__(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor
                 ):
        n_samples = y_pred.shape[0]
        log_likelihood = -torch.log(y_pred[range(n_samples), y_true])
        return torch.sum(log_likelihood) / n_samples

    def backward(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor
                 ):
        n_samples = y_pred.shape[0]
        softmax = Softmax()
        grad: torch.Tensor | torch.tensor = softmax(y_pred, dim=1)
        grad[range(n_samples), y_true] -= 1
        grad = grad / n_samples
        return grad

    def paramerters(self):
        return []


class Softmax:

    def __call__(self,
                 X: torch.Tensor,
                 dim: int):
        X = X - torch.max(X ,dim=1, keepdims=True).values
        sof = torch.exp(X) / torch.sum(torch.exp(X), dim=dim, keepdims=True)

    def parameters(self):
        return []


class OptimizerSG:

    def __init__(self,
                params: Optional[List],
                lr : float = 0.1):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class Flatten:

    def __call__(self,
                 X: torch.Tensor):
        self.X = X
        self.out = X.view(X.shape[0], -1)
        return self.out

    def backward(self,
                 dZ: torch.Tensor):
        dX = dZ.view(self.X.size())
        return dX

    def parameters(self):
        return []


class MaxPool2d:

    def __init__(self,
                 kernel_size: int | Tuple,
                 stride: int | Tuple):
        self.kernel_size = (kernel_size
                            if isinstance(kernel_size, tuple) and len(kernel_size) == 2
                            else (kernel_size, kernel_size)
        if isinstance(kernel_size, int) else (2, 2))
        self.stride = (stride
                       if isinstance(stride, tuple) and len(stride) == 2
                       else (stride, stride)
        if isinstance(stride, int) else (2, 2))
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride

    def prepare_submatrix(self, X: torch.Tensor):
        B, C, ih, iw = X.shape
        oh = (ih - self.kh) // self.sh + 1
        ow = (iw - self.kw) // self.sw + 1
        subM = X.unfold(2, self.kh, self.sh).unfold(3, self.kw, self.sw)
        return subM

    def forward(self, X: torch.Tensor):
        self.X = X
        subM = self.prepare_submatrix(X)
        return subM.max(dim=-1).values.max(dim=-1).values

    def add_padding(self,
                    X: torch.Tensor,
                    padding: int):
        padding = tuple(repeat(padding, 4))
        batch_size, in_channels, original_height, original_width = X.size()
        padded_height = original_height + padding[0] + padding[1]
        padded_width = original_width + padding[2] + padding[3]

        padded_x = torch.zeros((batch_size, in_channels, padded_height, padded_width), dtype=X.dtype)
        padded_x[:, :, padding[0]:padding[0] + original_height, padding[2]:padding[2] + original_width] = X
        return padded_x

    def prepare_mask(self,
                     subM: torch.Tensor,
                     kh: int,
                     kw: int
                     ):
        B, C, oh, ow, kh, kw = subM.shape
        print(torch.reshape(subM, (-1, kh * kw)))
        # a = subM.view(-1, kh * kw)
        a = torch.reshape(subM, (-1, kh * kw))
        idx = torch.argmax(a, dim=1)
        b = torch.zeros_like(a)
        b[torch.arange(b.shape[0]), idx] = 1
        mask = b.view(B, C, oh, ow, kh, kw)
        return mask

    def mask_dXp(self, mask: torch.Tensor,
                 Xp: torch.Tensor,
                 dz: torch.Tensor,
                 kh: int,
                 kw: int):
        dA = torch.einsum('i,ijk->ijk',
                          dz.view(-1),
                          mask.view(-1, kh, kw)).view(mask.shape)
        B, C, ih, iw = Xp.shape
        strides = (C * ih * iw, ih * iw, iw, 1)
        strides = tuple(i * Xp.element_size() for i in strides)
        dXp = torch.as_strided(dA, Xp.shape, strides)
        return dXp

    def parameters(Self):
        return []
