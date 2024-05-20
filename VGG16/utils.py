import torch


def set_seeds(manual_seed: int = 42):
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)


def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
