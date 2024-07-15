import torch


class Flatten:

    def __init__(self):
        self.X = None
        self.out = None
        self.output_shape = None
        self.Nw = None
        self.Nh = None
        self.Nc = None
        self.m = None

    def forward(self, X: torch.Tensor):
        self.m, self.Nc, self.Nh, self.Nw = X.shape
        self.X = X
        self.out = X.view(X.shape[0], -1)
        return self.out

    def backpropagation(self,
                        dZ: torch.Tensor):
        dX = dZ.view(self.X.size())
        return dX

    def get_dimensions(self, input_shape):

        if len(input_shape) == 4:
            self.m, self.Nc, self.Nh, self.Nw = input_shape
        elif len(input_shape) == 3:
            self.Nc, self.Nh, self.Nw = input_shape

        self.output_shape = self.Nc * self.Nh * self.Nw
