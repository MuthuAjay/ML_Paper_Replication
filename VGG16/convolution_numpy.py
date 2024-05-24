import numpy as np


# Helper functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    n_samples = y_pred.shape[0]
    res = y_pred[range(n_samples), y_true]
    return -np.sum(np.log(res)) / n_samples


def cross_entropy_loss_derivative(y_pred, y_true):
    n_samples = y_pred.shape[0]
    res = y_pred
    res[range(n_samples), y_true] -= 1
    return res / n_samples


# Layers
class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, 3, filter_size, filter_size) / 9

    def iterate_regions(self, image):
        h, w = image.shape[2], image.shape[3]
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[:, :, i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        batch_size, _, h, w = input.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((batch_size, self.num_filters, output_h, output_w))

        for im_region, i, j in self.iterate_regions(input):
            output[:, :, i, j] = np.tensordot(im_region, self.filters, axes=([1, 2, 3], [1, 2, 3]))

        return output

    def backward(self, d_L_d_out):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += np.sum(d_L_d_out[:, f, i, j][:, None, None, None] * im_region, axis=0)
        self.filters -= 0.001 * d_L_d_filters
        return d_L_d_filters


class MaxPoolLayer:
    def __init__(self, size):
        self.size = size

    def iterate_regions(self, image):
        h, w = image.shape[2], image.shape[3]
        new_h = h // self.size
        new_w = w // self.size
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[:, :, (i * self.size):(i * self.size + self.size),
                            (j * self.size):(j * self.size + self.size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape[2], input.shape[3]
        new_h = h // self.size
        new_w = w // self.size
        output = np.zeros((input.shape[0], input.shape[1], new_h, new_w))
        for im_region, i, j in self.iterate_regions(input):
            output[:, :, i, j] = np.amax(im_region, axis=(2, 3))
        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w = im_region.shape[2], im_region.shape[3]
            amax = np.amax(im_region, axis=(2, 3))
            for n in range(im_region.shape[0]):
                for f in range(im_region.shape[1]):
                    for i2 in range(h):
                        for j2 in range(w):
                            if im_region[n, f, i2, j2] == amax[n, f]:
                                d_L_d_input[n, f, i * self.size + i2, j * self.size + j2] = d_L_d_out[n, f, i, j]
        return d_L_d_input


class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_L_d_out):
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        d_L_d_input = d_L_d_input.reshape(self.last_input_shape)
        d_L_d_weights = np.dot(self.last_input[:, None], d_L_d_out[None, :])
        d_L_d_biases = d_L_d_out
        self.weights -= 0.001 * d_L_d_weights
        self.biases -= 0.001 * d_L_d_biases
        return d_L_d_input


# Model
class CNN:
    def __init__(self):
        self.conv = ConvLayer(8, 3)
        self.pool = MaxPoolLayer(2)
        # Adjusted input size for the fully connected layer
        self.fc = FCLayer(8 * 15 * 15, 128)
        self.output = FCLayer(128, 3)

    def forward(self, input):
        out = self.conv.forward(input)
        out = relu(out)
        out = self.pool.forward(out)
        out = self.fc.forward(out)
        out = relu(out)
        out = self.output.forward(out)
        return softmax(out)

    def train(self, x_train, y_train):
        out = self.forward(x_train)
        loss = cross_entropy_loss(out, y_train)
        print(f'Loss: {loss}')
        d_L_d_out = cross_entropy_loss_derivative(out, y_train)

        d_L_d_out = self.output.backward(d_L_d_out)
        d_L_d_out = self.fc.backward(d_L_d_out)
        d_L_d_out = self.pool.backward(d_L_d_out)
        d_L_d_out = relu_derivative(self.conv.backward(d_L_d_out))
        return loss


# Create random x_train and y_train
np.random.seed(42)  # For reproducibility
x_train = np.random.randn(10, 3, 32, 32)  # 10 samples, 3 channels, 32x32 images
y_train = np.random.randint(0, 3, size=(10,))  # 10 samples, 3 classes

# Example usage
cnn = CNN()
for epoch in range(10):  # Reduced epochs for example purposes
    loss = cnn.train(x_train, y_train)
