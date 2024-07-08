import numpy as np


class LearningRateDecay:

    def __init__(self):
        pass

    def constant(self, t, lr_0):
        '''
        t: iteration
        lr_0: initial learning rate
        '''
        return lr_0

    def time_decay(self, t, lr_0, k):
        '''
        lr_0: initial learning rate
        k: Decay rate
        t: iteration number
        '''
        lr = lr_0 / (1 + (k * t))
        return lr

    def step_decay(self, t, lr_0, F, D):
        '''
        lr_0: initial learning rate
        F: factor value controlling the rate in which the learning date drops
        D: “Drop every” iteration
        t: current iteration
        '''
        mult = F ** np.floor((1 + t) / D)
        lr = lr_0 * mult
        return lr

    def exponential_decay(self, t, lr_0, k):
        '''
        lr_0: initial learning rate
        k: Exponential Decay rate
        t: iteration number
        '''
        lr = lr_0 * np.exp(-k * t)
        return lr
