import torch
import torch.nn.functional as F


class Padding2D:

    def __init__(self, p='valid'):
        self.p = p

    def get_dimensions(self, input_shape, kernel_size, s=(1, 1)):
        if len(input_shape) == 4:
            m, Nc, Nh, Nw = input_shape
        elif len(input_shape) == 3:
            Nc, Nh, Nw = input_shape

        Kh, Kw = kernel_size
        sh, sw = s
        p = self.p

        if type(p) == int:
            pt, pb = p, p
            pl, pr = p, p

        elif type(p) == tuple:
            ph, pw = p
            pt, pb = ph // 2, (ph + 1) // 2
            pl, pr = pw // 2, (pw + 1) // 2

        elif p == 'valid':
            pt, pb = 0, 0
            pl, pr = 0, 0

        elif p == 'same':
            ph = max((Nh - 1) * sh + Kh - Nh, 0)
            pw = max((Nw - 1) * sw + Kw - Nw, 0)

            pt, pb = ph // 2, ph - ph // 2
            pl, pr = pw // 2, pw - pw // 2

        else:
            raise ValueError(
                "Incorrect padding type. Allowed types are only 'same', 'valid', an integer or a tuple of length 2.")

        if len(input_shape) == 4:
            output_shape = (m, Nc, Nh + pt + pb, Nw + pl + pr)
        elif len(input_shape) == 3:
            output_shape = (Nc, Nh + pt + pb, Nw + pl + pr)

        return output_shape, (pt, pb, pl, pr)

    def forward(self, X, kernel_size, s=(1, 1)):
        self.input_shape = X.shape
        m, Nc, Nh, Nw = self.input_shape

        self.output_shape, (self.pt, self.pb, self.pl, self.pr) = self.get_dimensions(self.input_shape,
                                                                                      kernel_size, s=s)

        Xp = F.pad(X, (self.pl, self.pr, self.pt, self.pb))
        return Xp

    def backpropagation(self, dXp):
        m, Nc, Nh, Nw = self.input_shape
        dX = dXp[:, :, self.pt:self.pt + Nh, self.pl:self.pl + Nw]
        return dX


class Pooling2D:

    def __init__(self, pool_size=(2, 2), s=(2, 2), p='valid', pool_type='max'):
        self.padding = Padding2D(p=p)

        if type(pool_size) == int:
            self.pool_size = (pool_size, pool_size)
        elif type(pool_size) == tuple and len(pool_size) == 2:
            self.pool_size = pool_size

        self.Kh, self.Kw = self.pool_size

        if type(s) == int:
            self.s = (s, s)
        elif type(s) == tuple and len(s) == 2:
            self.s = s

        self.sh, self.sw = self.s

        self.pool_type = pool_type

    def get_dimensions(self, input_shape):
        if len(input_shape) == 4:
            m, Nc, Nh, Nw = input_shape
        elif len(input_shape) == 3:
            Nc, Nh, Nw = input_shape

        Oh = (Nh - self.Kh) // self.sh + 1
        Ow = (Nw - self.Kw) // self.sw + 1

        if len(input_shape) == 4:
            self.output_shape = (m, Nc, Oh, Ow)
        elif len(input_shape) == 3:
            self.output_shape = (Nc, Oh, Ow)

    def prepare_subMatrix(self, X, pool_size, s):
        m, Nc, Nh, Nw = X.shape
        sh, sw = s
        Kh, Kw = pool_size

        Oh = (Nh - Kh) // sh + 1
        Ow = (Nw - Kw) // sw + 1

        subM = X.unfold(2, Kh, sh).unfold(3, Kw, sw)
        return subM

    def pooling(self, X, pool_size=(2, 2), s=(2, 2)):
        subM = self.prepare_subMatrix(X, pool_size, s)

        if self.pool_type == 'max':
            return subM.max(dim=-1).values.max(dim=-1).values
        elif self.pool_type == 'mean':
            return subM.mean(dim=(-1, -2))
        else:
            raise ValueError("Allowed pool types are only 'max' or 'mean'.")

    def prepare_mask(self, subM, Kh, Kw):
        m, Nc, Oh, Ow, Kh, Kw = subM.shape

        a = subM.view(-1, Kh * Kw)
        idx = torch.argmax(a, dim=1)
        b = torch.zeros_like(a)
        b[torch.arange(b.shape[0]), idx] = 1
        mask = b.view(m, Nc, Oh, Ow, Kh, Kw)

        return mask

    def mask_dXp(self, mask, Xp, dZ, Kh, Kw):
        dA = torch.einsum('i,ijk->ijk', dZ.view(-1), mask.view(-1, Kh, Kw)).view(mask.shape)
        m, Nc, Nh, Nw = Xp.shape
        strides = (Nc * Nh * Nw, Nh * Nw, Nw, 1)
        strides = tuple(i * Xp.element_size() for i in strides)
        dXp = torch.as_strided(dA, Xp.shape, strides)
        return dXp

    def maxpool_backprop(self, dZ, X):
        Xp = self.padding.forward(X, self.pool_size, self.s)
        subM = self.prepare_subMatrix(Xp, self.pool_size, self.s)

        m, Nc, Oh, Ow, Kh, Kw = subM.shape
        m, Nc, Nh, Nw = Xp.shape

        mask = self.prepare_mask(subM, Kh, Kw)
        dXp = self.mask_dXp(mask, Xp, dZ, Kh, Kw)

        return dXp

    def dZ_dZp(self, dZ):
        sh, sw = self.s
        Kh, Kw = self.pool_size

        dZp = dZ.repeat_interleave(Kh, dim=2).repeat_interleave(Kw, dim=3)

        jh, jw = Kh - sh, Kw - sw

        if jw != 0:
            L = dZp.shape[-1] - 1
            l1 = torch.arange(sw, L)
            l2 = torch.arange(sw + jw, L + jw)
            mask = torch.cat([torch.ones(jw), torch.zeros(jw)]).bool().repeat(len(l1) // jw)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, :, r1] += dZp[:, :, :, r2]
            dZp = torch.index_select(dZp, 3, torch.tensor(list(set(range(dZp.shape[3])) - set(r2.tolist()))))

        if jh != 0:
            L = dZp.shape[-2] - 1
            l1 = torch.arange(sh, L)
            l2 = torch.arange(sh + jh, L + jh)
            mask = torch.cat([torch.ones(jh), torch.zeros(jh)]).bool().repeat(len(l1) // jh)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, r1, :] += dZp[:, :, r2, :]
            dZp = torch.index_select(dZp, 2, torch.tensor(list(set(range(dZp.shape[2])) - set(r2.tolist()))))

        return dZp

    def averagepool_backprop(self, dZ, X):
        Xp = self.padding.forward(X, self.pool_size, self.s)
        m, Nc, Nh, Nw = Xp.shape

        dZp = self.dZ_dZp(dZ)

        ph = Nh - dZp.shape[-2]
        pw = Nw - dZp.shape[-1]

        padding_back = Padding2D(p=(ph, pw))

        dXp = padding_back.forward(dZp, s=self.s, kernel_size=self.pool_size)

        return dXp / (Nh * Nw)

    def forward(self, X):
        self.X = X
        Xp = self.padding.forward(X, self.pool_size, self.s)
        Z = self.pooling(Xp, self.pool_size, self.s)
        return Z

    def backpropagation(self, dZ):
        if self.pool_type == 'max':
            dXp = self.maxpool_backprop(dZ, self.X)
        elif self.pool_type == 'mean':
            dXp = self.averagepool_backprop(dZ, self.X)
        dX = self.padding.backpropagation(dXp)
        return dX
