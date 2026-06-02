import argparse
from types import SimpleNamespace

from attacks import EOTPGDL2, L2PGDAttack
from certify import certify_subset, save_certification_rows
from defenses import train_adversarial, train_gaussian, train_standard
from evaluate import accuracy
from model import SmallCNN
from utils import get_data, get_device, set_seed


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
    train_adversarial(cfg, l2_defended_model, train_loader, device)

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
