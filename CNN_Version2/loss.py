import torch
from activation import Softmax


class CrossEntropyLoss:

    def __call__(self,
                 y_preds: torch.Tensor,
                 y_true: torch.Tensor) -> torch.Tensor:
        n = y_preds.shape[0]
        log_likelihood = -torch.log(y_preds[range(0, n), y_true])
        self.out = torch.sum(log_likelihood) / n
        return self.out

    def backward(self,
                 y_pred: torch.Tensor,
                 y_true: torch.Tensor) -> torch.Tensor:
        n = y_pred.shape[0]
        softmax = Softmax()
        dz = softmax(y_pred)
        dz[range(0, n), y_true] -= 1
        return dz / n
