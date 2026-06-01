import argparse
import csv
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from attacks import EOTPGDL2, L2PGDAttack
from certify import ABSTAIN, _certify
from defenses import train_gaussian, train_standard
from model import SmallCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def get_data(dataset, data_dir, batch_size, train_limit=None, test_limit=None):
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

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, channels, num_classes, size


def accuracy(model, loader, device, attack=None):
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


def train_l2_adversarial(model, loader, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        attack = L2PGDAttack(
            model,
            epsilon=cfg.train_epsilon,
            k=cfg.train_attack_steps,
            a=cfg.train_step_size,
            random_start=True,
            loss_func="xent",
        )
        for x, y in tqdm(loader, desc=f"l2 defense train {epoch + 1}/{cfg.epochs}"):
            x, y = x.to(device), y.to(device)
            x_adv = attack.perturb(x, y)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()
        print(f"[l2_defense] epoch={epoch + 1}/{cfg.epochs} loss={total_loss / total:.4f}")


def certify_subset(model, loader, cfg, device, num_classes):
    xs, ys, seen = [], [], 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        seen += x.size(0)
        if seen >= cfg.certify_limit:
            break

    x_all = torch.cat(xs)[: cfg.certify_limit]
    y_all = torch.cat(ys)[: cfg.certify_limit]

    correct = 0
    certified = 0
    abstains = 0
    radius_sum = 0.0
    rows = []

    for i in tqdm(range(x_all.size(0)), desc="certify"):
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
        rows.append({"idx": i, "label": label, "prediction": pred, "radius": radius, "correct": is_correct})

    n = x_all.size(0)
    return {
        "certify_images": n,
        "smoothed_accuracy": correct / n,
        "abstention_rate": abstains / n,
        "mean_radius": radius_sum / n,
        f"certified_accuracy_at_{cfg.radius:g}": certified / n,
        "rows": rows,
    }


def save_certification_rows(path, rows):
    if not path:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "label", "prediction", "radius", "correct"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate standard, L2-PGD, and smoothing defenses.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)

    parser.add_argument("--epsilon", type=float, default=0.5, help="L2 PGD evaluation radius")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--train-epsilon", type=float, default=0.5)
    parser.add_argument("--train-attack-steps", type=int, default=7)
    parser.add_argument("--train-step-size", type=float, default=0.1)
    parser.add_argument("--eot-samples", type=int, default=8)

    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--certify-limit", type=int, default=100)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--certify-batch", type=int, default=128)
    parser.add_argument("--cert-csv", default="certification_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device={device} dataset={args.dataset}")

    train_loader, test_loader, channels, num_classes, size = get_data(
        args.dataset,
        args.data_dir,
        args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    standard_model = SmallCNN(channels, num_classes, size).to(device)
    l2_defended_model = SmallCNN(channels, num_classes, size).to(device)
    smoothed_model = SmallCNN(channels, num_classes, size).to(device)

    cfg = SimpleNamespace(**vars(args))

    print("\n=== train standard model ===")
    train_standard(cfg, standard_model, train_loader, device)

    print("\n=== train L2-PGD defended model ===")
    train_l2_adversarial(l2_defended_model, train_loader, cfg, device)

    print("\n=== train randomized smoothing base model ===")
    train_gaussian(cfg, smoothed_model, train_loader, device)

    standard_attack = L2PGDAttack(
        standard_model,
        epsilon=args.epsilon,
        k=args.attack_steps,
        a=args.step_size,
        random_start=True,
    )
    defended_attack = L2PGDAttack(
        l2_defended_model,
        epsilon=args.epsilon,
        k=args.attack_steps,
        a=args.step_size,
        random_start=True,
    )
    eot_attack = EOTPGDL2(
        smoothed_model,
        sigma=args.sigma,
        epsilon=args.epsilon,
        k=args.attack_steps,
        a=args.step_size,
        m=args.eot_samples,
        random_start=True,
    )

    print("\n=== accuracy results ===")
    results = {
        "standard_clean": accuracy(standard_model, test_loader, device),
        "standard_l2_pgd": accuracy(standard_model, test_loader, device, standard_attack),
        "l2_defended_clean": accuracy(l2_defended_model, test_loader, device),
        "l2_defended_l2_pgd": accuracy(l2_defended_model, test_loader, device, defended_attack),
        "smoothing_base_clean": accuracy(smoothed_model, test_loader, device),
        "smoothing_base_eot_l2_pgd": accuracy(smoothed_model, test_loader, device, eot_attack),
    }
    for key, value in results.items():
        print(f"{key:28s}: {value:.4f}")

    print("\n=== randomized smoothing certification ===")
    cert = certify_subset(smoothed_model, test_loader, cfg, device, num_classes)
    rows = cert.pop("rows")
    for key, value in cert.items():
        print(f"{key:28s}: {value:.4f}" if isinstance(value, float) else f"{key:28s}: {value}")
    save_certification_rows(args.cert_csv, rows)
    if args.cert_csv:
        print(f"\nwrote per-example certification rows to {args.cert_csv}")


if __name__ == "__main__":
    main()
