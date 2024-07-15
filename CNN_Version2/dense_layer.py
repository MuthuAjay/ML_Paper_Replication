import torch
from CNN_Version2.activation import *
from weights_initializer import WeightsInitializer
from optimizer import Optimizer


class Dense:

    def __init__(self,
                 out_features,
                 activation_type=None,
                 use_bias=True,
                 weight_initializer=None,
                 weight_regularizer=None,
                 seed=None,
                 input_dim=None):

        self.db = None
        self.X = None
        self.Z = None
        self.b = None
        self.W = None
        self.out_features = out_features
        if activation_type == "relu":
            self.activation = Relu()
        elif activation_type == 'softmax':
            self.activation = Softmax()
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        if weight_regularizer is None:
            self.weight_regularizer = ('L2', 0)
        else:
            self.weight_regularizer = weight_regularizer
        self.seed = seed
        self.input_dim = input_dim

    def initialize_parameters(self,
                              input_features,
                              optimizer_type):
        shape_w = (input_features, self.out_features)
        shape_b = (self.out_features, 1)
        initializer = WeightsInitializer(shape=shape_w,
                                         initializer_type=self.weight_initializer,
                                         seed=self.seed)
        self.W = initializer.get_initializer(mode='fan_out', non_linearity='relu')
        self.b = torch.zeros(shape_b)
        self.optimizer = Optimizer(optimizer_type=optimizer_type, shape_weights=shape_w, shape_bias=shape_b,
                                   )

    def forward(self,
                X: torch.Tensor):
        self.X = X
        r = X @ self.W
        self.Z = r + self.b.T
        a = self.activation(self.Z)
        return a

    def backpropagation(self,
                 da: torch.Tensor):
        dz = self.activation.backward(da)
        dr = dz.clone()
        self.db = torch.sum(dz, dim=0).view(-1, 1)
        self.dW = self.X.T @ dr
        dX = dr @ self.W.T
        return dX

    def update(self, lr, m, k):
        dW, db = self.optimizer.get_optimization(self.dW, self.db, k)
        if self.weight_regularizer[0].lower == "l2":
            dW += self.weight_regularizer[1] * self.W
        elif self.weight_regularizer[0].lower == 'l1':
            dW += self.weight_regularizer[1] * self.W

        self.W -= dW * (lr / m)
        if self.use_bias:
            self.b -= db * (lr / m)
