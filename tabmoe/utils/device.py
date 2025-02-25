import os
import torch


def get_device() -> torch.device:
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_gpu_names() -> list[str]:
    return [
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ]

def is_dataparallel_available() -> bool:
    return (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and 'CUDA_VISIBLE_DEVICES' in os.environ
    )