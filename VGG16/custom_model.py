import torch
import torch.nn.functional as F
from itertools import repeat


# Helper functions
def relu(x):
    return torch.clamp(x, min=0)


def relu_derivative(x):
    return (x > 0).float()


def softmax(x):
    exps = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return exps / torch.sum(exps, dim=1, keepdim=True)


def cross_entropy_loss(y_pred, y_true):
    n_samples = y_pred.shape[0]
    log_likelihood = -torch.log(y_pred[range(n_samples), y_true])
    return torch.sum(log_likelihood) / n_samples


def cross_entropy_loss_derivative(y_pred, y_true):
    n_samples = y_pred.shape[0]
    grad = y_pred.clone()
    grad[range(n_samples), y_true] -= 1
    grad = grad / n_samples
    return grad


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


# Max Pool Layer
class MaxPoolLayer:
    def __init__(self, size):
        self.size = size

    def forward(self, x):
        self.last_input = x
        batch_size, channels, height, width = x.shape
        output_height = height // self.size
        output_width = width // self.size
        output = torch.zeros((batch_size, channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                region = x[:, :, i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size]
                print(region.shape)
                output[:, :, i, j] = torch.max(region.view(batch_size, channels, -1), dim=2)[0]

        return output

    def backward(self, d_L_d_out):
        d_L_d_input = torch.zeros_like(self.last_input)

        for i in range(d_L_d_out.shape[2]):
            for j in range(d_L_d_out.shape[3]):
                region = self.last_input[:, :, i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size]
                max_region = \
                    torch.max(region.view(self.last_input.shape[0], self.last_input.shape[1], -1), dim=2, keepdim=True)[
                        0]
                mask = (region == max_region.unsqueeze(-1))
                d_L_d_input[:, :, i * self.size:(i + 1) * self.size,
                j * self.size:(j + 1) * self.size] += mask * d_L_d_out[:, :, i, j].unsqueeze(-1).unsqueeze(-1)

        return d_L_d_input


# Fully Connected Layer
class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = torch.randn(input_size, output_size, requires_grad=True) / input_size
        self.biases = torch.zeros(output_size, requires_grad=True)

    def forward(self, x):
        self.last_input_shape = x.shape
        self.last_input = x.view(x.shape[0], -1)
        return torch.matmul(self.last_input, self.weights) + self.biases

    def backward(self, d_L_d_out, lr):
        d_L_d_weights = torch.matmul(self.last_input.t(), d_L_d_out)
        d_L_d_biases = torch.sum(d_L_d_out, dim=0)

        d_L_d_input = torch.matmul(d_L_d_out, self.weights.t())
        self.weights = self.weights - lr * d_L_d_weights
        self.biases = self.biases - lr * d_L_d_biases

        return d_L_d_input


# Model
class CNN:
    def __init__(self):
        self.conv = Conv2d(3, 8, 3)
        self.pool = MaxPoolLayer(2)
        self.fc = FCLayer(8 * 15 * 15, 128)
        self.output = FCLayer(128, 3)

    def forward(self, x):
        x = self.conv(x)
        x = relu(x)
        x = self.pool.forward(x)
        x = self.fc.forward(x)
        x = relu(x)
        x = self.output.forward(x)
        return softmax(x)

    def train(self, x_train, y_train, lr=0.001):
        out = self.forward(x_train)
        loss = cross_entropy_loss(out, y_train)
        print(f'Loss: {loss.item()}')

        grad = cross_entropy_loss_derivative(out, y_train)

        grad = self.output.backward(grad, lr)
        grad = relu_derivative(self.fc.backward(grad, lr))
        grad = self.pool.backward(grad)
        grad = relu_derivative(self.conv.backward(grad, lr))

        return loss


# Create random x_train and y_train
torch.manual_seed(42)
x_train = torch.randn(10, 3, 32, 32)  # 10 samples, 3 channels, 32x32 images
y_train = torch.randint(0, 3, (10,))  # 10 samples, 3 classes

# Example usage
cnn = CNN()
for epoch in range(10):  # Reduced epochs for example purposes
    loss = cnn.train(x_train, y_train)
