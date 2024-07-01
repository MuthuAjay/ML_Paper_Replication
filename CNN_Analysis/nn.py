import torch
from typing import List, Tuple
from conv_layer import Conv2d
from activations import Relu
from itertools import repeat


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

    def __init__(self, kernel_size: int | Tuple[int, int], stride: int | Tuple[int, int]):
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = tuple(stride) if isinstance(stride, tuple) else (stride, stride)
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        self.padded_height, self.padded_width = None, None

    def prepare_submatrix(self, X: torch.Tensor):
        B, C, ih, iw = X.shape
        oh = (ih - self.kh) // self.sh + 1
        ow = (iw - self.kw) // self.sw + 1
        subM = X.unfold(2, self.kh, self.sh).unfold(3, self.kw, self.sw)
        return subM

    def __call__(self, X: torch.Tensor):
        self.X = X
        subM = self.prepare_submatrix(X)
        self.out = subM.max(dim=-1).values.max(dim=-1).values
        return self.out

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

    def prepare_mask(self, subM: torch.Tensor):
        B, C, oh, ow, kh, kw = subM.shape
        a = torch.reshape(subM, (-1, kh * kw))
        idx = torch.argmax(a, dim=1)
        b = torch.zeros_like(a)
        b[torch.arange(b.shape[0]), idx] = 1
        mask = b.view(B, C, oh, ow, kh, kw)
        return mask

    def mask_dXp(self, mask: torch.Tensor, dz: torch.Tensor):
        dz_expanded = dz.unsqueeze(-1).unsqueeze(-1).expand_as(mask)
        dXp = dz_expanded * mask
        return dXp

    def maxpool_backprop(self, dZ: torch.Tensor, X: torch.Tensor):
        Xp = self.add_padding(X, self.kernel_size[0])
        subM = self.prepare_submatrix(Xp)
        mask = self.prepare_mask(subM)
        dXp = self.mask_dXp(mask, dZ)
        return dXp

    def padding_backward(self, dXp: torch.Tensor):
        B, C, ih, iw = self.X.shape
        dX = dXp[:, :, self.padded_height:ih, self.padded_width:iw]
        return dX

    def backward(self, dL_dout):
        Batch, num_channels, input_height, input_width = self.X.shape
        dL_dinput = torch.zeros_like(self.X)
        output_height = (input_height - self.kh) // self.sh + 1
        output_width = (input_width - self.kw) // self.sw + 1

        # Extract patches from the input tensor
        subM = self.prepare_submatrix(self.X)

        # Create the mask for the max pooling operation
        mask = self.prepare_mask(subM)

        # Expand dL_dout to match the shape of mask and perform element-wise multiplication
        dL_dout_expanded = dL_dout.unsqueeze(-1).unsqueeze(-1).expand_as(mask)
        dL_dinput_unfolded = dL_dout_expanded * mask

        # Combine the unfolded gradients to form the final gradient
        dL_dinput = dL_dinput_unfolded.contiguous().view(Batch, num_channels, output_height, output_width, self.kh,
                                                         self.kw)
        dL_dinput = dL_dinput.permute(0, 1, 2, 4, 3, 5).contiguous().view(Batch, num_channels, output_height * self.kh,
                                                                          output_width * self.kw)

        # Reduce the overlapping areas by summing them
        result = torch.zeros_like(self.X)
        for i in range(self.kh):
            for j in range(self.kw):
                result[:, :, i::self.kh, j::self.kw] += dL_dinput[:, :, i::self.kh, j::self.kw]

        return result

    def parameters(self):
        return []


class Linear:
    def __init__(self,
                 fan_in: int,
                 fan_out: int,
                 bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.randn(fan_out) if bias else None

    def __call__(self,
                 X: torch.Tensor):
        self.last_input = X
        self.out = X @ self.weight.T
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def backward(self, d_L_d_out):
        # d_L_d_weights = torch.matmul(self.last_input.t(), d_L_d_out)

        d_L_d_weights = self.last_input.T @ d_L_d_out
        d_L_d_biases = torch.sum(d_L_d_out, dim=0)
        d_L_d_input = d_L_d_out @ self.weight

        return d_L_d_input, d_L_d_weights, d_L_d_biases

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class Sequential:
    def __init__(self,
                 layers: List):
        self.layers = layers

    def __call__(self,
                 X: torch.Tensor):
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def backward(self, logits):
        dlogits = loss.backward(logits, y)
        grads = []
        for layer in self.layers[::-1]:
            if isinstance(layer, Conv2d):
                dlogits, dconv_w, dconv_b = layer.backward(dlogits)
                grads += [dconv_b, dconv_w]
                layer.weights.grad = dconv_w
                layer.bias.grad = dconv_b
            elif isinstance(layer, Relu):
                dlogits = layer.backward(dlogits)
                layer.out.grad = dlogits
            elif isinstance(layer, MaxPool2d):
                dlogits = layer.backward(dlogits)
            elif isinstance(layer, Flatten):
                dlogits = layer.backward(dlogits)
            elif isinstance(layer, Linear):
                dlogits, d_L_d_weights, d_L_d_bias = layer.backward(dlogits)
                grads += [d_L_d_bias, d_L_d_weights]
                layer.weight.grad = d_L_d_weights.T
                layer.bias.grad = d_L_d_bias
        return grads[::-1]
