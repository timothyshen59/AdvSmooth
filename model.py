"""
Model definitions L2-robust classifiers.

Currently provides SmallCNN, a lightweight convolutional network
used as the base classifier for randomized smoothing experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SmallCNN(nn.Module):
    def __init__(self, input_channel: int, num_classes: int, size: int):
        """
        Args:
            input_channel: Number of input channels (e.g. 1 for MNIST, 3 for CIFAR).
            num_classes: Number of output classes.
            size: Spatial size of the input (assumed square, e.g. 28 for MNIST).
        """
        super().__init__()
        conv_layers = [
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        ]

        dummy_input = torch.zeros(1, input_channel, size, size)
        with torch.no_grad():
            flattened_dim = nn.Sequential(*conv_layers)(dummy_input).flatten(1).shape[1]

        self.net = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            nn.Linear(flattened_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.net(x)
