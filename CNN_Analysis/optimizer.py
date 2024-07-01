from typing import Optional, List


class OptimizerSG:

    def __init__(self,
                 params: Optional[List],
                 lr: float = 0.1):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
