from typing import Tuple
import torch
import torch.nn.functional as F
from itertools import repeat


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
        return (torch.randn(self.out_channels, self.in_channels // self.groups,
                            *self.kernel_size),
                torch.zeros(self.out_channels))

    def add_padding(self,
                    x: torch.Tensor,
                    padding: int):
        padding = tuple(repeat(padding, 4))
        batch_size, in_channels, original_height, original_width = x.size()
        padded_height = original_height + padding[0] + padding[1]
        padded_width = original_width + padding[2] + padding[3]

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
        # Input (N, Cin, Hin, Win)
        batch_size, in_channels, in_height, in_width = x.size()
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // \
                     self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[
            1] + 1

        if self.padding[0] > 0 or self.padding[1] > 0:
            x = self.add_padding(x, self.padding[0])

        # Initialize the output
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width)

        # Perform convolution
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride[0]
                h_end = h_start + self.kernel_size[0]
                w_start = w * self.stride[1]
                w_end = w_start + self.kernel_size[1]
                receptive_field = x[:, :, h_start:h_end, w_start:w_end]

                # Element-wise multiplication and summation
                out[:, :, h, w] = torch.sum(
                    receptive_field.unsqueeze(1) * self.weights.view(1, self.out_channels,
                                                                     self.in_channels // self.groups,
                                                                     *self.kernel_size),
                    dim=(2, 3, 4)
                ) + self.bias.view(1, self.out_channels)

        return out


# Example usage
x = torch.randn(3, 3, 5, 5)  # Batch size of 1, 3 input channels, 5x5 image
conv = Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
output = conv(x)
print(output.shape)  # Should be (1, 1, 5, 5)
