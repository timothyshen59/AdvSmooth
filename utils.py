"""
Utility functions for reproducibility, device selection, and data loading.
"""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Covers CUDA, MPS, and cuDNN backends if available.

    Args:
        seed: Random seed value, defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if getattr(torch, "mps", None) is not None:
        torch.mps.manual_seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolve and return the appropriate torch device.

    Args:
        requested: Device string. 'auto' selects CUDA, then MPS, then CPU.
                   Otherwise passed directly to torch.device().

    Returns:
        Selected torch.device.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def get_data(
    dataset: str,
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, int, int, int]:
    """
    Load a dataset and return train and test DataLoaders with metadata.

    Args:
        dataset: Dataset name, either 'mnist' or 'cifar10'.
        data_dir: Directory to download or load data from.
        batch_size: Batch size for both loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, test_loader, channels, num_classes, size)
        where size is the spatial dimension of the input (assumed square).

    Raises:
        ValueError: If dataset is not 'mnist' or 'cifar10'.
    """
    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_split = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_split = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        channels, num_classes, size = 1, 10, 28
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_split = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_split = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        channels, num_classes, size = 3, 10, 32
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, channels, num_classes, size