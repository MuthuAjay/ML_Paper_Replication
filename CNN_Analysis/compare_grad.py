import torch


# Utility function which we will use later when copamring Manual Gradients to Pytorch Gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approximate {str(app):5s} | maxdiff: {maxdiff}")
