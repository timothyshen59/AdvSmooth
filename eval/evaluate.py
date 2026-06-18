"""
Evaluation utilities for clean and adversarial accuracy.

Provides two evaluation functions:
    - accuracy: simple clean or adversarial top-1 accuracy.
    - accuracy_with_class_conf: top-1 accuracy with per-class confidence,
      supporting both standard and smoothed classifiers.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm



def accuracy(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    attack: Any | None = None,
) -> float:
    """
    Compute top-1 accuracy over a dataloader, optionally under attack.

    Args:
        model: The classifier to evaluate.
        loader: DataLoader providing (input, label) batches.
        device: Device to run evaluation on.
        attack: Optional attack object with a perturb(x, y) method.
                If None, evaluates on clean inputs.

    Returns:
        Top-1 accuracy as a float between 0 and 1.
    """
    
    model.eval()
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="accuracy", leave=False):
        x, y = x.to(device), y.to(device)
        if attack is not None:
            x = attack.perturb(x, y)
        with torch.no_grad():
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def accuracy_with_class_conf(
    model: Module,
    loader: DataLoader,
    device: torch.device,
    attack: Any | None = None,
    num_classes: int = 10,
    smoothed: bool = False,
    sigma: float = 0.25,
    n_smooth: int = 64,
) -> tuple[float, list[float]]:
    """
    Compute top-1 accuracy and per-class confidence, optionally under attack.

    When smoothed=True, estimates class probabilities by averaging softmax
    outputs over n_smooth noisy samples with standard deviation sigma,
    matching the noise convention used in training and certification.

    Args:
        model: The classifier to evaluate.
        loader: DataLoader providing (input, label) batches.
        device: Device to run evaluation on.
        attack: Optional attack object with a perturb(x, y) method.
        num_classes: Total number of classes.
        smoothed: If True, evaluate the smoothed classifier g by averaging
                  over noisy samples. 
                  If False, evaluate the base classifier f.
        sigma: Standard deviation of Gaussian noise (only used if smoothed=True).
        n_smooth: Number of noisy samples to average over (only used if smoothed=True).

    Returns:
        Tuple of (overall_accuracy, per_class_confidence) where
        per_class_confidence is a list of length num_classes containing
        the mean predicted probability for the true class per class.
    """
    
    model.eval()
    correct = 0
    total = 0

    confidence_sum = torch.zeros(num_classes)
    class_cnt = torch.zeros(num_classes)

    for x, y in tqdm(loader, desc="accuracy", leave=False):
        x, y = x.to(device), y.to(device)
        if attack is not None:
            x = attack.perturb(x, y)  # attack g (EOT) -- before scoring

        with torch.no_grad():
            if smoothed:
                # g classsifer
                probs = torch.zeros(x.size(0), num_classes, device=device)
                for _ in range(n_smooth):
                    noise = (
                        torch.randn_like(x) * sigma
                    )  # same noise convention as training/certify
                    probs += F.softmax(model(x + noise), dim=1)
                probs /= n_smooth
            else:
                # f classifier
                probs = F.softmax(model(x), dim=1)

            pred = probs.argmax(1)
            true_prob = probs.gather(1, y.view(-1, 1)).squeeze(1)

        correct += (pred == y).sum().item()
        total += y.numel()

        for c in range(num_classes):
            mask = y == c
            if mask.any():
                confidence_sum[c] += true_prob[mask].sum().item()
                class_cnt[c] += mask.sum().item()

    per_class_confidence = (confidence_sum / class_cnt.clamp(min=1)).tolist()
    return correct / total, per_class_confidence
