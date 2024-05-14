from typing import List, Dict, Union, Optional, Tuple
from itertools import repeat
import torch
import torch.nn.functional as F


class Conv2d:
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weights, self.bias = self.initialise_weights()

    def _n_tuple(self):
        for p in [self.kernel_size, self.stride,
                  self.padding, self.dilation]:
            if isinstance(p, Tuple):
                continue
            else:
                p = tuple([p, p])

    def initialise_weights(self):
        return (torch.randn(self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size),
                torch.zeros(self.out_channels)
                )

    def __call__(self, x: torch.Tensor):
        # Input (N, Cin, Win, Hin) or (Cin, Win, Hin)
        # Output (N, Cout, Wout, Hout) or (Cout, Wout, Hout)
        batch_size, in_channels, in_height, in_width = x.size()


class Conv2dCustom:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = torch.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, in_height, in_width = x.size()
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Pad input if necessary
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Initialize output tensor
        out = torch.zeros(batch_size, self.out_channels, out_height, out_width)

        # Perform convolution
        for h in range(out_height):
            for w in range(out_width):
                h_start = h * self.stride
                h_end = h_start + self.kernel_size
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                receptive_field = x[:, :, h_start:h_end, w_start:w_end]

                # Element-wise multiplication and summation
                out[:, :, h, w] = torch.sum(
                    receptive_field * self.weight.view(1, self.out_channels, self.in_channels, self.kernel_size,
                                                       self.kernel_size), dim=(2, 3, 4)) + self.bias.view(1,
                                                                                                          self.out_channels)

        return out


# Test the implementation
x = torch.randn(1, 1, 5, 5)  # Batch size of 1, 1 channel, 5x5 input
conv = Conv2dCustom(1, 1, 3, padding=1)  # 1 input channel, 1 output channel, 3x3 kernel, padding 1
output = conv.forward(x)
print(output.size())
