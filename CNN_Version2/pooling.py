import torch
from typing import Tuple
from padding import Padding


class Poolin2D:
    def __init__(self, pool_size=(2, 2), s=(2, 2), p="valid", pool_type="max"):

        self.Kw = None
        self.Kh = None
        self.output_shape = None
        self.padding = Padding(p=p)

        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        elif isinstance(pool_size, Tuple) and len(pool_size) == 2:
            self.pool_size = pool_size

        self.kh, self.kw = self.pool_size

        if isinstance(s, int):
            self.s = (s, s)
        elif isinstance(s, Tuple) and len(s) == 2:
            self.s = s

        self.sh, self.sw = self.s

        self.pool_type = pool_type

    def get_dimensions(self, input_shape):

        if len(input_shape) == 4:
            B, Nc, Nh, Nw = input_shape
        elif len(input_shape) == 3:
            Nc, Nh, Nw = input_shape
        else:
            raise ValueError("Invalid input")
        Oh = (Nh - self.Kh) // self.sh + 1
        Ow = (Nw - self.Kw) // self.sw + 1

        if len(input_shape) == 4:
            self.output_shape = (B, Nc, Oh, Ow)
        elif len(input_shape) == 3:
            self.output_shape = (Nc, Oh, Ow)

    def prepare_subMatrix(self, X: torch.Tensor, pool_size, s):
        B, C, Nh, Nw = X.shape
        sh, sw = s
        kh, kw = pool_size
        oh = (Nh - kh) // sh + 1
        ow = (Nw - kw) // sw + 1
        subM = X.unfold(2, kh, sh).unfold(3, kw, sw)
        return subM

    def pooling(self, X, pool_size=(2, 2), s=(2, 2)):

        subM = self.prepare_subMatrix(X, pool_size, s)

        if self.pool_type == 'max':
            return subM.max(dim=-1).values.max(dim=-1).values
        elif self.pool_type == 'mean':
            return subM.mean(dim=-1).mean(dim=-1)
        else:
            raise ValueError("Allowed pool types are only 'max' or 'mean'.")

    def prepare_mask(self, subM: torch.Tensor, kh, kw):
        B, C, oh, ow, kh, kw = subM.shape
        a = torch.reshape(subM, (-1, kh * kw))
        idx = torch.argmax(a, dim=1)
        b = torch.zeros_like(a)
        b[torch.arange(b.shape[0]), idx] = 1
        mask = b.view(B, C, oh, ow, kh, kw)
        return mask

    def mask_dXp(self, mask: torch.Tensor, dz: torch.Tensor) -> object:
        dz_expanded = dz.unsqueeze(-1).unsqueeze(-1).expand_as(mask)
        dXp = dz_expanded * mask
        return dXp

    def maxpool_backprop(self, dZ: torch.Tensor, X: torch.Tensor):
        Xp = self.padding.forward(X, self.pool_size, self.s)
        subM = self.prepare_subMatrix(Xp, self.pool_size, self.s)

        B, Nc, Oh, Ow, Kh, Kw = subM.shape
        mask = self.prepare_mask(subM, Kh, Kw)
        dXp = self.mask_dXp(mask, dZ)

        return dXp

    def averagepool_backprop(self, dZ, x):
        return ''

    def forward(self, X):
        '''
        Parameters:

        X: input of shape (m, Nc, Nh, Nw)

        Returns:

        Z: pooled X
        '''

        self.X = X

        # padding
        Xp = self.padding.forward(X, self.pool_size, self.s)

        Z = self.pooling(Xp, self.pool_size, self.s)

        return Z

    def backpropagation(self, dZ):
        '''
        Parameters:

        dZ: Output Error

        Return:

        dX: Backprop Error of X
        '''
        if self.pool_type == 'max':
            dXp = self.maxpool_backprop(dZ, self.X)
        elif self.pool_type == 'mean':
            dXp = self.averagepool_backprop(dZ, self.X)
        else:
            raise ValueError("Invalid Pool Type")
        dX = self.padding.backward(dXp)
        return dX
