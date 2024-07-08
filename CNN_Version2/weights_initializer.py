import math
import torch


class WeightsInitializer:

    def __init__(self,
                 shape,
                 initializer_type=None,
                 seed=None):
        self.shape = shape
        if initializer_type is None:
            self.initializer_type = "he_normal"
        else:
            self.initializer_type = initializer_type

        self.seed = seed

    def kaiming_uniform(self, tensor: torch.Tensor,
                        a=0, mode="fan_in", non_linearity='leaky_relu'):
        fan = self.calculate_fan(tensor, mode)
        gain = self.calculate_gain(non_linearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def calculate_fan(self,
                      tensor: torch.Tensor,
                      mode: str = "fan_in"):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and Fan out cannot be computed for the tensors with fewer than 2 dimensions")

        elif dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_inputs_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = tensor[0][0].numel()
            fan_in = num_inputs_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size

        return fan_in if mode == 'fan_in' else fan_out

    def calculate_gain(self, non_linearity, param=None):
        linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d',
                      'conv_transpose3d']

        if non_linearity in linear_fns or non_linearity == "sigmoid":
            return 1
        elif non_linearity == "tanh":
            return 5.0 / 3
        elif non_linearity == "relu":
            return math.sqrt(2.0)
        elif non_linearity == "leaky_relu":
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
                negative_slope = 0.01
            else:
                raise ValueError("negative slope {} not a valid number".format(param))
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        else:
            raise ValueError("Unsupported NonLinearity {}".format(non_linearity))

    def he_initializer(self, weights: torch.Tensor,
                       mode="fan_in", non_linearity='leaky_relu'):
        self.kaiming_uniform(weights)

    def get_initializer(self, weights: torch.Tensor, mode="fan_in", non_linearity='leaky_relu'):
        if self.initializer_type == "he_normal":
            return self.he_initializer(weights, mode, non_linearity)
