"""
Randomized smoothing certification (Cohen et al., 2019).
Certifies L2 robustness radii for a smoothed classifier by Monte Carlo sampling.
"""

import csv
from typing import Any

import torch
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

ABSTAIN = -1  # Sentinel value for certification confidence under threshold


@torch.no_grad()
def _noise_counts(
    model: Module,
    x: Tensor,
    n: int,
    sigma: float,
    num_classes: int,
    batch_size: int,
) -> Tensor:
    """
    Sample noisy predictions and return per-class counts.
    Adds Gaussian noise with standard deviation sigma to x across n samples,
    runs the model on each, and counts how many times each class was predicted.

    Args:
        model: The base classifier.
        x: Input tensor of shape (1, C, H, W).
        n: Number of noisy samples to draw.
        sigma: Standard deviation of the Gaussian noise.
        num_classes: Total number of classes.
        batch_size: Number of samples to process per forward pass.

    Returns:
        Integer tensor of shape (num_classes,) with per-class prediction counts.
    """
    counts = torch.zeros(num_classes, dtype=torch.long, device=x.device)
    remaining = n

    while remaining > 0:
        this_batch_size = min(batch_size, remaining)
        remaining -= this_batch_size

        batch = x.repeat((this_batch_size, 1, 1, 1))
        noise = torch.randn_like(batch) * sigma

        predictions = model(batch + noise).argmax(1)
        counts += torch.bincount(predictions, minlength=num_classes)

    return counts


def _certify(
    model: Module,
    x: Tensor,
    sigma: int,
    n0: int,
    n: int,
    alpha: float,
    num_classes: int,
    batch_size: int,
) -> tuple[int, float]:
    """
    Run the Cohen et al. certification procedure for a single input.

    Args:
        model: The base classifier to smooth.
        x: Input tensor of shape (1, C, H, W).
        sigma: Noise standard deviation.
        n0: Number of samples for selecting the top class.
        n: Number of samples for estimating pA.
        alpha: Failure probability.
        num_classes: Total number of classes.
        batch_size: Inference batch size.

    Returns:
        (predicted_class, certified_radius), or (ABSTAIN, 0.0) if pA_bar < 0.5.
    """
    model.eval()

    c0 = _noise_counts(model, x, n0, sigma, num_classes, batch_size)

    c_hat = c0.argmax().item()

    c = _noise_counts(model, x, n, sigma, num_classes, batch_size)

    nA = c[c_hat].item()
    pA_bar = proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0]

    if pA_bar < 0.5:
        return ABSTAIN, 0.0

    return c_hat, float(sigma * norm.ppf(pA_bar))


def _certify_limit(cfg: Any) -> int:
    limit = getattr(cfg, "certify_limit", None)
    if limit is None:
        limit = getattr(cfg, "n_certify", None)
    if limit is None:
        raise AttributeError(
            "certification config must define certify_limit or n_certify"
        )
    return limit


def certify_subset(
    model: Module,
    loader: DataLoader,
    cfg: Any,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """
    Run certification over the first certify_limit samples from loader.

    For each sample, calls _certify to obtain a predicted class and certified
    L2 radius, then aggregates accuracy, abstention rate, and certified accuracy
    at the radius threshold defined in cfg.

    Args:
        model: The base classifier to smooth.
        loader: DataLoader providing (input, label) batches.
        cfg: Config obj. (see run_resutls.py)
        device: Device for inference
        num_classes: Total # of classes 

    Returns:
        Dict containing:
            - certify_images: Number of samples certified.
            - smoothed_accuracy: Fraction of correct predictions.
            - abstention_rate: Fraction of samples where certification abstained.
            - mean_radius: Average certified radius across all samples.
            - certified_accuracy_at_{r}: Fraction correct and certified at radius r.
            - rows: Per-sample list of dicts with idx, label, prediction, radius, correct.
    """
    xs, ys, seen = [], [], 0
    limit = _certify_limit(cfg)

    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= limit:
            break

    x_all = torch.cat(xs)[:limit]
    y_all = torch.cat(ys)[:limit]

    correct = 0
    certified = 0
    abstains = 0
    radius_sum = 0.0
    rows = []

    for i in range(x_all.size(0)):
        x_i = x_all[i].unsqueeze(0).to(device)
        pred, radius = _certify(
            model,
            x_i,
            cfg.sigma,
            cfg.n0,
            cfg.n,
            cfg.alpha,
            num_classes,
            cfg.certify_batch,
        )
        label = int(y_all[i])
        is_correct = pred == label
        is_certified = is_correct and radius >= cfg.radius

        correct += int(is_correct)
        certified += int(is_certified)
        abstains += int(pred == ABSTAIN)
        radius_sum += radius
        rows.append(
            {
                "idx": i,
                "label": label,
                "prediction": pred,
                "radius": radius,
                "correct": is_correct,
            }
        )

    n = x_all.size(0)
    return {
        "certify_images": n,
        "smoothed_accuracy": correct / n,
        "abstention_rate": abstains / n,
        "mean_radius": radius_sum / n,
        f"certified_accuracy_at_{cfg.radius:g}": certified / n,
        "rows": rows,
    }


def save_certification_rows(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["idx", "label", "prediction", "radius", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)
