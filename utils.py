import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int = 42):
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


def get_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def get_data(dataset, data_dir="./data", batch_size=128, train_limit=None, test_limit=None, num_workers=0):
    if dataset == "mnist":
        transform = transforms.ToTensor()
        train_split = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_split = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        channels, num_classes, size = 1, 10, 28
    elif dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )
        train_split = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_split = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        channels, num_classes, size = 3, 10, 32
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    if train_limit is not None:
        train_split = Subset(train_split, range(min(train_limit, len(train_split))))
    if test_limit is not None:
        test_split = Subset(test_split, range(min(test_limit, len(test_split))))

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, channels, num_classes, size
