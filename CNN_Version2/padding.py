import torch
from typing import Tuple


class Padding:

    def __init__(self, p="valid"):
        self.output_shape = None
        self.input_shape = None
        self.p = p

    def get_dimensions(self, input_shape, kernel_size, s=(1, 1)):

        if len(input_shape) == 4:
            B, Nc, Nh, Nw = input_shape
        elif len(input_shape) == 3:
            Nc, Nh, Nw = input_shape

        kh, kw = kernel_size
        sh, sw = s
        p = self.p

        if isinstance(p, int):
            pt, pb = p, p
            pl, pr = p, p

        if isinstance(p, Tuple):
            ph, pw = p
            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2

        elif p == "valid":
            pt, pb, pl, pr = 0, 0, 0, 0

        elif p == "same":
            ph = (sh - 1) * Nh + kh - sh
            pw = (sw - 1) * Nw + kw - sw

            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2
        else:
            raise ValueError("Incorrect padding type. Allowed types are only 'same' , 'valid' , 'int' and 'tuple'")

        if len(input_shape) == 4:
            output_shape = (B, Nc, Nh + pt + pb, Nw + pl + pr)
        elif len(input_shape) == 3:
            output_shape = (Nc, Nh + pt + pb, Nw + pl + pr)

        return output_shape, (pt, pb, pl, pr)

    def forward(self, X: torch.Tensor, kernel_size, s=(1, 1)):
        self.input_shape = X.shape
        B, Nc, Nh, Nw = self.input_shape

        self.output_shape, (self.pt, self.pb, self.pl, self.pr) = self.get_dimensions(self.input_shape, kernel_size, s)
        zeros_r = torch.zeros((B, Nc, Nh, self.pr))
        zeros_l = torch.zeros((B, Nc, Nh, self.pl))
        zeros_t = torch.zeros((B, Nc, self.pt, Nw + self.pl + self.pr))
        zeros_b = torch.zeros((B, Nc, self.pb, Nw + self.pl + self.pr))

        Xp = torch.concat((X, zeros_r), dim=3)
        Xp = torch.concat((zeros_l, Xp), dim=3)
        Xp = torch.concat((zeros_t, Xp), dim=2)
        Xp = torch.concat((Xp, zeros_b), dim=2)

        return Xp

    def backward(self, dXp: torch.Tensor):
        B, Nc, Nh, Nw = self.input_shape
        dX = dXp[:, :, self.pt:self.pt + Nh, self.pl:self.pl + Nw]
        return dX
