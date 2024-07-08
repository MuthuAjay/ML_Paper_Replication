import torch


class Optimizer:

    def __init__(self,
                 optimizer_type=None,
                 shape_weights=None,
                 shape_bias=None,
                 momentum1=0.9,
                 momentum2=0.99,
                 epsilon=1e-8):

        if optimizer_type is None:
            self.optimizer_type = 'adam'
        else:
            self.optimizer_type = optimizer_type

        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.epsilon = epsilon

        self.vdw = torch.zeros(shape_weights)
        self.vdb = torch.zeros(shape_bias)

        self.Sdw = torch.zeros(shape_weights)
        self.Sdb = torch.zeros(shape_bias)

    def SGD(self, dw, db, k):

        self.vdw = self.momentum1 * self.vdw + (1 - self.momentum1) * dw
        self.vdb = self.momentum1 * self.vdb + (1 - self.momentum1) * db

        return self.vdw, self.vdb

    def Adam(self, dw, db, k):

        # momentum
        self.vdw = self.momentum1 * self.vdw + (1 - self.momentum1) * dw
        self.vdb = self.momentum1 * self.vdb + (1 - self.momentum1) * db

        # rmsprop
        self.Sdw = self.momentum2 * self.Sdw + (1 - self.momentum2) * (dw ** 2)
        self.Sdb = self.momentum2 * self.Sdb + (1 - self.momentum2) * (db ** 2)

        # correction
        if k > 1:
            vdw_h = self.vdw / (1 - (self.momentum1 ** k))
            vdb_h = self.vdb / (1 - (self.momentum1 ** k))
            Sdw_h = self.Sdw / (1 - (self.momentum2 ** k))
            Sdb_h = self.Sdb / (1 - (self.momentum2 ** k))
        else:
            vdw_h = self.vdw
            vdb_h = self.vdb
            Sdw_h = self.Sdw
            Sdb_h = self.Sdb

        den_w = torch.sqrt(Sdw_h) + self.epsilon
        den_b = torch.sqrt(Sdb_h) + self.epsilon

        return vdw_h / den_w, vdb_h / den_b
