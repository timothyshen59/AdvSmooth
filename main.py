"""
Main entry point for training and evaluating defended models.

Supports three training modes (standard, adversarial, Gaussian noise augmentation)
and two evaluation modes (clean, adversarial). Optionally runs randomized smoothing
certification for smoothing-trained models and logs all results to a CSV file.

Usage:
    python train_eval.py --dataset mnist --smooth-training --eval-attack eot
    python train_eval.py --dataset cifar10 --adv-training --adv-attack pgd_l2
    python train_eval.py --load-model --run-name my_run --eval-attack none
"""

import argparse
from types import SimpleNamespace
from typing import Any 

import torch

from attacks.attacks import EOTPGDL2, L2PGDAttack
from eval.certify import certify_subset, save_certification_rows
from results.csv_utils import append_run_csv
from defenses.defenses import train_adversarial, train_gaussian, train_standard
from eval.evaluate import accuracy_with_class_conf
from model import SmallCNN
from utils import get_data, get_device, set_seed

from torch.utils.data import DataLoader
from torch.nn import Module

def make_attack(
    attack_type: str,
    model: torch.nn.Module,
    epsilon: float,
    attack_steps: int,
    step_size: float,
    sigma: float,
    m: int,
):
    """
    Instantiates L2 attack object by type 
    
    Args:
        attack_type: Either 'pgd_l2' for standard PGD or 'eot' for
                     EOT-PGD against smoothed classifiers.
        model: The classifier to attack.
        epsilon: L2 perturbation budget.
        attack_steps: Number of PGD iterations.
        step_size: Step size per iteration.
        sigma: Gaussian noise standard deviation (only used if attack_type='eot').
        m: Number of noise samples to average over (only used if attack_type='eot').

    Returns:
        Configured attack object ready to call .perturb(x, y) on.

    Raises:
        ValueError: If attack_type is not 'pgd_l2' or 'eot'.
    """
    
    if attack_type == "pgd_l2":
        print("\n===L2 PGD Attack  ===")

        output_attack = L2PGDAttack(
            model,
            epsilon=epsilon,
            k=attack_steps,
            a=step_size,
            random_start=True,
        )
    elif attack_type == "eot":
        print("\n===PGD EOT Attack  ===")

        output_attack = EOTPGDL2(
            model,
            epsilon=epsilon,
            k=attack_steps,
            a=step_size,
            random_start=True,
            sigma=sigma,
            m=m,
        )

    else:
        raise ValueError(f"unknown attack kind: {attack_type}")

    return output_attack


def train_single(config: Any, model: torch.nn.Module, loader: DataLoader, device: torch.device)->None:
    """
    Train the model using the mode specified in config.

    Training mode is determined by config.adv_training and config.smooth_training:
        - Both False:  standard cross-entropy training.
        - adv_training only:  adversarial training with PGD-L2 or EOT-PGD.
        - smooth_training only:  Gaussian noise augmentation for randomized smoothing.
        - Both True:  adversarial training followed by Gaussian noise augmentation.

    Args:
        config: Config object with attributes adv_training, smooth_training,
                and any fields required by the respective training function.
        model: The classifier to train.
        loader: DataLoader providing (input, label) batches.
        device: Device to run training on.
    """

    if config.adv_training:
        print("\n===Adversarially Training against attack ===")
        train_adversarial(config, model, loader, device)
    if config.smooth_training:
        print("\n=== Randomized Smooth Training for Model ===")
        train_gaussian(config, model, loader, device)

    if not config.adv_training and not config.smooth_training:
        print("\n=== Standard Training for Model ===")
        train_standard(config, model, loader, device)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single defended model."
    )
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)

    # --- defenses ---
    parser.add_argument("--adv-training", action="store_true")
    parser.add_argument(
        "--adv-attack",
        choices=["pgd_l2", "eot"],
        default="pgd_l2",
        help="attack used during adversarial training",
    )
    parser.add_argument("--smooth-training", action="store_true")
    parser.add_argument(
        "--eval-attack",
        choices=["none", "pgd_l2", "eot"],
        default="eot",
        help="attack applied at evaluation; 'none' = clean run",
    )

    parser.add_argument("--epsilon", type=float, default=1.0, help="L2 perturbation budget")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.125)
    parser.add_argument("--train-epsilon", type=float, default=1.0)
    parser.add_argument("--train-attack-steps", type=int, default=20)
    parser.add_argument("--train-step-size", type=float, default=0.125)
    parser.add_argument("--eot-samples", type=int, default=16)

    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--certify-limit", type=int, default=2000)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--certify-batch", type=int, default=128)
    parser.add_argument("--cert-csv", default="certification_rows.csv")
    parser.add_argument("--results-csv", default="runs.csv", help="append one row per run")

    parser.add_argument("--run-name", default="eps1.0-m-at-rs", help="name used for checkpoint file")
    parser.add_argument("--training", action="store_true", help="run training before evaluation")
    parser.add_argument("--load-model", action="store_true", help="load model from checkpoint before evaluation")
    parser.add_argument("--checkpoint", action="store_true", help="save model checkpoint after training")

    return parser


def _run_evaluation(
    model: Module,
    test_loader: DataLoader,
    cfg: Any,
    device: torch.device,
    num_classes: int,
    eval_attack: Any | None,
) -> tuple[float, list[float]]:
    """Evaluate accuracy and per-class confidence on the test set."""
    acc, per_class_conf = accuracy_with_class_conf(
        model,
        test_loader,
        device,
        attack=eval_attack,
        num_classes=num_classes,
        smoothed=cfg.smooth_training,
    )
    mean_conf = sum(per_class_conf) / len(per_class_conf)
    print(f"{'accuracy':24s}: {acc:.4f}  (eval_attack={cfg.eval_attack})")
    print(f"{'mean true-label conf':24s}: {mean_conf:.4f}")
    print(
        f"{'per-class confidence':24s}: "
        + ", ".join(f"{c}:{v:.3f}" for c, v in enumerate(per_class_conf))
    )
    return acc, per_class_conf


def _build_record(
    cfg: Any,
    acc: float,
    per_class_conf: list[float],
) -> dict[str, Any]:
    """Assemble the results dict for CSV logging."""
    mean_conf = sum(per_class_conf) / len(per_class_conf)
    record: dict[str, Any] = {
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "epochs": cfg.epochs,
        "randomized_smoothing": str(cfg.smooth_training).lower(),
        "pgd_l2_defense": str(cfg.adv_training).lower(),
        "adv_attack_type": (cfg.adv_attack if cfg.adv_training else "none"),
        "eval_attack": cfg.eval_attack,
        "epsilon": cfg.epsilon,
        "sigma": cfg.sigma,
        "accuracy": acc,
        "mean_confidence": mean_conf,
    }
    for c, v in enumerate(per_class_conf):
        record[f"conf_class_{c}"] = v
    return record


def _run_certification(
    model: Module,
    test_loader: DataLoader,
    cfg: Any,
    device: torch.device,
    num_classes: int,
    record: dict[str, Any],
) -> None:
    """Run randomized smoothing certification and update record in place."""
    print("\n=== Randomized smoothing certification (clean inputs) ===")
    cert = certify_subset(model, test_loader, cfg, device, num_classes)
    rows = cert.pop("rows")
    for key, value in cert.items():
        print(
            f"{key:28s}: {value:.3f}"
            if isinstance(value, float)
            else f"{key:28s}: {value}"
        )
    save_certification_rows(cfg.cert_csv, rows)
    record["cert_smoothed_accuracy"] = cert.get("smoothed_accuracy")
    record["cert_abstention_rate"] = cert.get("abstention_rate")
    record["cert_mean_radius"] = cert.get("mean_radius")
    record["cert_radius"] = cfg.radius
    record["cert_certified_accuracy"] = cert.get(f"certified_accuracy_at_{cfg.radius:g}")


def main() -> None:
    """
    Train and evaluate a single defended model.

    Parses CLI arguments, optionally trains and checkpoints the model,
    evaluates accuracy under a clean or adversarial eval, and runs
    randomized smoothing certification if smooth training is enabled.
    Results are appended to a CSV file.
    """
    args = _build_parser().parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print(
        f"device={device} dataset={args.dataset} seed={args.seed} "
        f"adv={'on(' + args.adv_attack + ')' if args.adv_training else 'off'} "
        f"smooth={'on' if args.smooth_training else 'off'} eval_attack={args.eval_attack}"
    )

    train_loader, test_loader, channels, num_classes, size = get_data(
        args.dataset, args.data_dir, args.batch_size
    )
    model = SmallCNN(channels, num_classes, size).to(device)
    cfg = SimpleNamespace(**vars(args))

    if cfg.training:
        print("\n===Training===")
        train_single(cfg, model, train_loader, device)

    if cfg.training and cfg.checkpoint:
        print(f"\nSaving model to {cfg.run_name}")
        torch.save(model.state_dict(), f"checkpoints/{cfg.run_name}.pth")

    if cfg.load_model:
        print(f"Loading model from {cfg.run_name}")
        model.load_state_dict(torch.load(f"checkpoints/{cfg.run_name}.pth", map_location=device))

    eval_attack = None
    if cfg.eval_attack != "none":
        eval_attack = make_attack(
            cfg.eval_attack, model, cfg.epsilon,
            cfg.attack_steps, cfg.step_size, cfg.sigma, cfg.eot_samples,
        )

    print("\n=== Accuracy + per-class confidence ===")
    acc, per_class_conf = _run_evaluation(model, test_loader, cfg, device, num_classes, eval_attack)
    record = _build_record(cfg, acc, per_class_conf)

    # certification only applies to smoothing-trained models
    if cfg.smooth_training:
        _run_certification(model, test_loader, cfg, device, num_classes, record)

    append_run_csv(cfg.results_csv, record, num_classes)
    print(f"\nAppended run to {cfg.results_csv}")


if __name__ == "__main__":
    main()