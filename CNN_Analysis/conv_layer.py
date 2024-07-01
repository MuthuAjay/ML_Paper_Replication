import torch
import math
from typing import Tuple


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = calculate_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def calculate_fan(tensor, mode='fan_in'):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in if mode == 'fan_in' else fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


class Conv2d:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        self.output_shape = None
        self.Ow = None
        self.Oh = None
        self.iw = None
        self.ih = None
        self.C = None
        self.B = None
        self.input_shape = None
        self.input_shape_x = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kh, self.kw = self.kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.sh, self.sw = self.stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weights, self.bias = self.initialise_parameters()

    def initialise_parameters(self, bias: bool = True):
        weights = torch.empty(self.out_channels, self.in_channels, *self.kernel_size, requires_grad=True)
        weights = kaiming_uniform(weights, mode='fan_out', nonlinearity='relu')
        bias = torch.zeros(self.out_channels, requires_grad=True) if not bias else torch.randn(self.out_channels,
                                                                                               requires_grad=True)

        return weights, bias

    def get_padding_dimensions(self,
                               input_shape: torch.Tensor.size,
                               kernel_size: Tuple,
                               s=(1, 1),
                               padding: int | Tuple = None):
        if len(input_shape) == 4:
            B, C, ih, iw = input_shape
        if len(input_shape) == 3:
            C, ih, iw = input_shape

        kh, kw = kernel_size
        sh, sw = s
        if padding is None:
            p = self.padding
        else:
            p = padding

        if isinstance(p, int):
            pt, pb, pl, pr = p, p, p, p
        elif isinstance(p, tuple):
            ph, pw = p
            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2
        elif p == 'valid':
            pt, pb = 0, 0
            pl, pr = 0, 0

        elif p == 'same':
            ph = (sh - 1) * ih + kh - sh
            pw = (sw - 1) * iw + kw - sw

            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2
        else:
            raise ValueError(
                "Incorrect padding type. Allowed types are only 'same', 'valid', an integer or a tuple of length 2.")

        if len(input_shape) == 4:
            output_shape = (B, C, ih + pt + pb, iw + pl + pr)
        elif len(input_shape) == 4:
            output_shape = (C, ih + pt + pb + iw + pl + pr)

        return output_shape, (pt, pb, pl, pr)

    def get_dimensions(self, input_shape: torch.Tensor):
        self.input_shape_x = input_shape.shape
        self.input_shape, _ = self.get_padding_dimensions(self.input_shape_x, self.kernel_size, self.stride)

        if len(self.input_shape) == 3:
            self.C, self.ih, self.iw = self.input_shape
        elif len(self.input_shape) == 4:
            self.B, self.C, self.ih, self.iw = self.input_shape

        self.Oh = (self.ih - self.kh) // self.sh + 1
        self.Ow = (self.iw - self.kw) // self.sw + 1

        if len(self.input_shape) == 3:
            self.output_shape = (self.out_channels, self.Oh, self.Ow)
        elif len(self.input_shape) == 4:
            self.output_shape = (self.B, self.out_channels, self.Oh, self.Ow)

    def prepare_subMatrix(self, X: torch.Tensor, Kh: int, Kw: int, s):
        B, C, ih, iw = X.shape
        sh, sw = s

        Oh = (ih - Kh) // sh + 1
        Ow = (iw - Kw) // sw + 1

        strides = (C * ih * iw, iw * ih, iw * sh, sw, iw, 1)
        subM = torch.as_strided(X,
                                size=(B, C, Oh, Ow, Kh, Kw),
                                stride=strides)
        return subM

    def convolve(self, X: torch.Tensor, K: torch.Tensor, s: Tuple = (1, 1), mode: str = 'back'):
        F, Kc, Kh, Kw = K.shape
        subM = self.prepare_subMatrix(X, Kh, Kw, s)
        if mode == 'front':
            return torch.einsum('fckl,bcijkl->bfij', K, subM)
        elif mode == 'back':
            return torch.einsum('fdkl,bcijkl->bdij', K, subM)
        elif mode == 'param':
            return torch.einsum('bfkl,bcijkl->fcij', K, subM)

    def padding_forward(self, X: torch.Tensor, kernel_size, s=(1, 1), padding=None) -> torch.Tensor:
        self.input_shape_before_padding = X.shape
        B, C, ih, iw = self.input_shape_before_padding
        self.output_shape_padded, (self.pt, self.pb, self.pl, self.pr) = self.get_padding_dimensions(
            self.input_shape_before_padding, kernel_size, s, padding=padding)

        zeros_r = torch.zeros((B, C, ih, self.pr), dtype=X.dtype, device=X.device)
        zeros_l = torch.zeros((B, C, iw, self.pl), dtype=X.dtype, device=X.device)
        zeros_t = torch.zeros((B, C, self.pt, iw + self.pl + self.pr), dtype=X.dtype, device=X.device)
        zeros_b = torch.zeros((B, C, self.pb, iw + self.pl + self.pr), dtype=X.dtype, device=X.device)

        Xp = torch.concat((X, zeros_r), dim=3)
        Xp = torch.concat((zeros_l, Xp), dim=3)
        Xp = torch.concat((zeros_t, Xp), dim=2)
        Xp = torch.concat((Xp, zeros_b), dim=2)

        return Xp

    def padding_backward(self, dXp: torch.Tensor):
        B, C, ih, iw = self.input_shape
        dX = dXp[:, :, self.pt:self.pt + ih, self.pl:self.pl + iw]
        return dX

    def dilate2D(self, X: torch.Tensor, Dr=(1, 1)) -> torch.Tensor:
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

    def dZ_D_dX(self, dZ_D: torch.Tensor, ih: int, iw: int) -> torch.Tensor:
        _, _, Hd, Wd = dZ_D.shape
        ph = ih - Hd + self.kh - 1
        pw = iw - Wd + self.kw - 1

        dZ_Dp = self.padding_forward(dZ_D, self.kernel_size, self.stride, (ph, pw))
        k_rotated = self.weights.flip([2, 3])
        dXp = self.convolve(dZ_Dp, k_rotated, mode='back')
        dX = self.padding_backward(dXp)

        return dX

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        self.X = X
        self.get_dimensions(X)
        Xp = self.padding_forward(X, self.kernel_size, self.stride, self.padding)
        self.Z = self.convolve(Xp, self.weights, self.stride, mode='front')

        if self.bias is not None:
            self.Z += self.bias.view(1, -1, 1, 1)  # sum should be done on the last layer
            self.out = self.Z.clone()
            return self.out
        self.out = self.Z.clone()
        return self.out

    def backward(self, dZ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Xp = self.padding_forward(self.X, self.kernel_size, self.stride)

        B, C, ih, iw = Xp.shape

        # Dilate dZ (dZ -> dZ_D)
        dZ_D = self.dilate2D(dZ, Dr=self.stride)
        dX = self.dZ_D_dX(dZ_D, ih, iw)

        # Gradient K k=kernel=weights=w
        _, _, Hd, Wd = dZ_D.shape

        ph = self.ih - Hd - self.kh + 1
        pw = self.iw - Wd - self.kw + 1

        dZ_Dp = self.padding_forward(dZ_D, self.kernel_size, self.stride, padding=(ph, pw))
        # self.dw = self.convolve(dZ_Dp, Xp, mode='param')
        self.dw = self.convolve(Xp, dZ_Dp, mode='param')

        # gradient db
        self.db = torch.sum(dZ, dim=[0, 2, 3])

        return dX, self.dw, self.db

    def parameters(self):
        return [self.weights] + ([] if self.bias is None else [self.bias])
