import torch
from typing import Tuple
from padding import Padding
from activation import *
from weights_initializer import WeightsInitializer
from optimizer import Optimizer


class Conv2D:

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int | Tuple | str = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 use_bias: bool = True,
                 activation_type=None,
                 weight_initializer_type=None,
                 kernel_regularizer=None,
                 seed=None,
                 input_shape=None
                 ):
        self.X = None
        self.b = None
        self.k = None
        self.output_shape = None
        self.Ow = None
        self.Oh = None
        self.Nw = None
        self.Nh = None
        self.Nc = None
        self.B = None
        self.input_shape = None
        self.in_channels = in_channels
        self.padding = Padding(p=padding)
        self.out_channels = out_channels

        self.input_shape_x = input_shape

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, Tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size

        self.kh, self.kw = self.kernel_size

        if isinstance(stride, int):
            self.s = (stride, stride)
        elif isinstance(stride, Tuple) and len(stride) == 2:
            self.s = stride

        self.sh, self.sw = self.s
        self.activation_type = activation_type
        if self.activation_type == "relu":
            self.activation = Relu()
        self.use_bias = use_bias
        self.weight_initializer_type = weight_initializer_type
        if kernel_regularizer is None:
            self.kernel_regularizer = ('L2', 0)
        else:
            self.kernel_regularizer = kernel_regularizer

        self.seed = seed
        self.dilate = dilation
        self.groups = groups

    def get_dimensions(self, input_shape):

        self.input_shape_x = input_shape
        self.input_shape, _ = self.padding.get_dimensions(self.input_shape_x,
                                                          kernel_size=self.kernel_size,
                                                          s=self.s)
        if len(self.input_shape) == 4:
            self.B, self.Nc, self.Nh, self.Nw = self.input_shape
        elif len(self.input_shape) == 3:
            self.Nc, self.Nh, self.Nw = self.input_shape

        self.Oh = (self.Nh - self.kh) // self.sh + 1
        self.Ow = (self.Nw - self.kw) // self.sw + 1

        self.output_shape = (self.B, self.out_channels, self.Oh, self.Ow) if len(self.input_shape) == 4 else (
            self.out_channels, self.Oh, self.Ow)

    def initialize_parameters(self, input_shape, optimizer_type):

        self.get_dimensions(input_shape)
        shape_b = (self.out_channels, self.Oh, self.Ow)
        shape_k = (self.out_channels, self.Nc, self.kh, self.kw)

        initializer = WeightsInitializer(shape=shape_k,
                                         initializer_type=self.weight_initializer_type,
                                         seed=self.seed)
        self.k = initializer.get_initializer(mode='fan_out',
                                             non_linearity=self.activation_type)

        self.b = torch.zeros(shape_b)
        self.optimizer = Optimizer(optimizer_type=optimizer_type,
                                   shape_weights=shape_k,
                                   shape_bias=shape_b)

    def dilate2D(self, X: torch.Tensor,
                 Dr: Tuple = (1, 1)):
        dh, dw = Dr
        B, C, H, W = X.shape

        if dw > 1:
            Xd_w = torch.zeros((B, C, H, W + (W - 1) * (dw - 1)), dtype=X.dtype, device=X.device)
            Xd_w[:, :, :, ::dw] = X
        else:
            Xd_w = X

        if dh > 1:
            Xd_h = torch.zeros((B, C, H + (H - 1) * (dh - 1), Xd_w.shape[-1]), dtype=X.dtype, device=X.device)
            Xd_h[:, :, ::dh, :] = Xd_w
        else:
            Xd_h = Xd_w

        return Xd_h

    def prepare_subMatrix(self, X: torch.Tensor, Kh: int, Kw: int, s):
        B, C, Nh, Nw = X.shape
        sh, sw = s

        Oh = (Nh - Kh) // sh + 1
        Ow = (Nw - Kw) // sw + 1

        strides = (C * Nh * Nw, Nw * Nh, Nw * sh, sw, Nw, 1)
        subM = torch.as_strided(X,
                                size=(B, C, Oh, Ow, Kh, Kw),
                                stride=strides)
        return subM

    def convolve(self, X: torch.Tensor, K: torch.Tensor, s: Tuple = (1, 1), mode: str = 'back'):
        B, Kc, Kh, Kw = K.shape
        subM = self.prepare_subMatrix(X, Kh, Kw, s)
        if mode == 'front':
            return torch.einsum('fckl,bcijkl->bfij', K, subM)
        elif mode == 'back':
            return torch.einsum('fdkl,bcijkl->bdij', K, subM)
        elif mode == 'param':
            return torch.einsum('bfkl,bcijkl->fcij', K, subM)

    def dZ_D_dX(self, dZ_D: torch.Tensor, ih: int, iw: int) -> torch.Tensor:
        _, _, Hd, Wd = dZ_D.shape
        ph = ih - Hd + self.kh - 1
        pw = iw - Wd + self.kw - 1

        padding_back = Padding(p=(ph, pw))
        dZ_Dp = padding_back.forward(dZ_D, self.kernel_size, self.s)
        k_rotated = self.k.flip([2, 3])
        dXp = self.convolve(dZ_Dp, k_rotated, mode='back')
        dX = self.padding.backward(dXp)

        return dX

    def forward(self, X):
        # padding

        self.X = X

        Xp = self.padding.forward(X, self.kernel_size, self.s)

        # convolve Xp with K
        Z = self.convolve(Xp, self.k, self.s) + self.b

        a = self.activation(Z)

        return a

    def backpropagation(self, da):

        Xp = self.padding.forward(self.X, self.kernel_size, self.s)

        m, Nc, Nh, Nw = Xp.shape

        dZ = self.activation.backward(da)

        # Dilate dZ (dZ-> dZ_D)

        dZ_D = self.dilate2D(dZ, Dr=self.s)

        dX = self.dZ_D_dX(dZ_D, Nh, Nw)

        # Gradient dK

        _, _, Hd, Wd = dZ_D.shape

        ph = self.Nh - Hd - self.kh + 1
        pw = self.Nw - Wd - self.kw + 1

        padding_back = Padding(p=(ph, pw))

        dZ_Dp = padding_back.forward(dZ_D, self.kernel_size, self.s)

        self.dK = self.convolve(Xp, dZ_Dp, mode='param')

        # Gradient db

        self.db = torch.sum(dZ, dim=0)

        return dX

    def update(self, lr, m, k):
        '''
        Parameters:

        lr: learning rate
        m: batch_size (sumber of samples in batch)
        k: iteration_number
        '''
        dK, db = self.optimizer.get_optimization(self.dK, self.db, k)

        if self.kernel_regularizer[0].lower() == 'l2':
            dK += self.kernel_regularizer[1] * self.K
        elif self.weight_regularizer[0].lower() == 'l1':
            dK += self.kernel_regularizer[1] * torch.sign(self.K)

        self.k -= self.dK * (lr / m)

        if self.use_bias:
            self.b -= self.db * (lr / m)