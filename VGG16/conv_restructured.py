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
        X = X - torch.max(X, dim=1, keepdims=True).values
        sof = torch.exp(X) / torch.sum(torch.exp(X), dim=dim, keepdims=True)

    def parameters(self):
        return []


class OptimizerSG:

    def __init__(self,
                 params: Optional[List],
                 lr: float = 0.1):
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


class Linear:
    def __init__(self,
                 fan_in: int,
                 fan_out: int,
                 bias=True):
        self.weight = torch.randn((fan_in, fan_out)) // fan_in ** 0.5
        self.bias = torch.randn(fan_out) if bias else None

    def __call__(self,
                 X: torch.Tensor):
        self.last_input = X
        self.out = X @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def backward(self, d_L_d_out, lr):
        # d_L_d_weights = torch.matmul(self.last_input.t(), d_L_d_out)

        d_L_d_weights = self.last_input.T @ d_L_d_out
        d_L_d_biases = torch.sum(d_L_d_out, dim=0)
        d_L_d_input = d_L_d_out @ self.weights.T

        return d_L_d_input

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# Custom Conv2d Layer
class Conv2d:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._check_parameters()
        self._n_tuple()
        self.weights, self.bias = self.initialise_weights()

    def _n_tuple(self):
        self.kernel_size = (self.kernel_size, self.kernel_size)
        self.stride = (self.stride, self.stride)
        self.padding = (self.padding, self.padding)
        self.dilation = (self.dilation, self.dilation)

    def initialise_weights(self):
        return (torch.randn(self.out_channels, self.in_channels // self.groups, *self.kernel_size, requires_grad=True),
                torch.zeros(self.out_channels, requires_grad=True))

    def add_padding(self, x: torch.Tensor, padding: int):
        padding = tuple(repeat(padding, 4))
        batch_size, in_channels, original_height, original_width = x.size()
        padded_height = original_height + padding[0] + padding[1]
        padded_width = original_width + padding[2] + padding[3]

        if (self.padded_height and self.padded_width) is None:
            self.padded_height, self.padded_width = padded_height, padded_width

        padded_x = torch.zeros((batch_size, in_channels, padded_height, padded_width), dtype=x.dtype)
        padded_x[:, :, padding[0]:padding[0] + original_height, padding[2]:padding[2] + original_width] = x
        return padded_x

    def _check_parameters(self):
        if self.groups <= 0:
            raise ValueError('groups must be a positive integer')
        if self.in_channels % self.groups != 0:
            raise ValueError('in_channels should be divisible by groups')
        if self.out_channels % self.groups != 0:
            raise ValueError('out_channels should be divisible by groups')

    def __call__(self, x: torch.Tensor):
        self.X = x
        self.ih, self.iw = self.X.shape[-2], self.X.shape[-1]
        batch_size, in_channels, in_height, in_width = x.size()
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // \
                     self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[
            1] + 1

        if self.padding[0] > 0 or self.padding[1] > 0:
            x = self.add_padding(x, self.padding[0])

        out = torch.zeros(batch_size, self.out_channels, out_height, out_width)

        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                receptive_field = x[:, :, h_start:h_end, w_start:w_end]

                out[:, :, h, w] = torch.sum(
                    receptive_field.unsqueeze(1) * self.weights.view(1, self.out_channels,
                                                                     self.in_channels // self.groups,
                                                                     *self.kernel_size),
                    dim=(2, 3, 4)
                ) + self.bias.view(1, self.out_channels)

        return out

    def prepare_subMatrix(self, X, Kh, Kw, s):
        B, C, ih, iw = X.shape
        sh, sw = s

        Oh = (ih - Kh) // sh + 1
        Ow = (iw - Kw) // sw + 1

        strides = (C * ih * iw, iw * ih, iw * sh, sw, iw, 1)
        subM = torch.as_strided(X,
                                shape=(B, C, Oh, Ow, Kh, Kw),
                                strides=strides
                                )
        return subM

    def padding_backward(self,
                         dXp: torch.Tensor):

        B, C, ih, iw = self.X.shape
        dX = dXp[:, :, self.padded_height:ih, self.padded_width:iw]
        return dX

    def convolve(self, X: torch.Tensor,
                 K: torch.Tensor,
                 s: Tuple = (1, 1),
                 mode: str = 'back'):

        F, Kc, Kh, Kw = K.shape
        subM = self.prepare_subMatrix(X, Kh, Kw, s)

        if mode == 'front':
            return torch.einsum('fckl,mcijkl->mfij', K, subM)
        elif mode == 'back':
            return torch.einsum('fdkl,mcijkl->mdij', K, subM)
        elif mode == 'param':
            return torch.einsum('mfkl,mcijkl->fcij', K, subM)

    def dz_D_dx(self,
                dZ: torch.Tensor,
                ih: int,
                iw: int):
        _, _, Hd, Wd = dZ.shape
        ph = ih - Hd + self.kernel_size[0] - 1
        pw = iw - Wd + self.kernel_size[0] - 1

        dZ_Dp = self.add_padding(dZ, ph)

        # Rotate the Kernel by 180 degrees
        k_rotated = self.weights[:, :, ::-1, ::-1]

        # convolve w.r.t k_rotated
        dXp = self.convolve(dZ_Dp, k_rotated, mode='back')

        dX = self.padding_backward(dXp)
        return dX

    def backward(self,
                 dZ: torch.Tensor):
        Xp: torch.Tensor = self.add_padding(self.X, self.padding)
        B, C, ih, iw = Xp.shape

        # dZ -> dZ_D_dX
        dX = self.dz_D_dx(dZ, ih, iw)

        # gradient dK
        _, _, Hd, Wd = dZ.shape
        ph = self.ih - Hd - self.kernel_size[0] + 1
        pw = self.iw - Wd - self.kernel_size[0] + 1

        dZ_Dp = self.add_padding(dZ, padding=ph)

        self.dweights = self.convolve(Xp, dZ_Dp, mode='param')

        # gradient db
        self.dbias = dZ.sum(0)

        return dX

    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])
