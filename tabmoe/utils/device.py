import torch


def get_device() -> torch.device:
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
