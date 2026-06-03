import argparse
from types import SimpleNamespace

from attacks import EOTPGDL2, L2PGDAttack
from certify import certify_subset, save_certification_rows
from defenses import train_adversarial, train_gaussian, train_standard
from evaluate import accuracy_with_class_conf
from model import SmallCNN
from utils import get_data, get_device, set_seed
from results_csv import append_run_csv

def make_attack(attack_type, model, epsilon, attack_steps, step_size, sigma, m): 
    if attack_type == "pgd_l2": 
        output_attack =  L2PGDAttack(
            model,
            epsilon=epsilon,
            k=attack_steps,
            a=step_size,
            random_start=True,
        )
    elif attack_type == "eot": 
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


def train_single(config, model, loader, device): 
    if config.adv_training: 
        print("\n===Adversarially Training against attack ===")
        train_adversarial(config, model, loader, device)
    if config.smooth_training: 
        print("\n=== Randomized Smooth Training for Model ===")
        train_gaussian(config, model, loader, device)

    else: 
        print("\n=== Standard Training for Model ===")
        train_standard(config, model, loader, device)

    

  
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a single defended model.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="cifar10")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--test-limit", type=int, default=1000)
 
    # --- defenses (the flags train_single reads) ---
    parser.add_argument("--adv-training", action="store_true")
    parser.add_argument("--adv-attack", choices=["pgd_l2", "eot"], default="pgd_l2",
                        help="attack used during adversarial training (takes effect in the combined branch)")
    parser.add_argument("--smooth-training", action="store_true")
    parser.add_argument("--eval-attack", choices=["none", "pgd_l2", "eot"], default="eot",
                        help="attack applied to inputs at evaluation; 'none' = clean run")
 
    parser.add_argument("--epsilon", type=float, default=0.1, help="L2 PGD evaluation radius")
    parser.add_argument("--attack-steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.0625)
    parser.add_argument("--train-epsilon", type=float, default=0.5)
    parser.add_argument("--train-attack-steps", type=int, default=20)
    parser.add_argument("--train-step-size", type=float, default=0.0625)
    parser.add_argument("--eot-samples", type=int, default=8)
 
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--certify-limit", type=int, default=500)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--n0", type=int, default=100)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--certify-batch", type=int, default=128)
    parser.add_argument("--cert-csv", default="certification_rows.csv")
    parser.add_argument("--results-csv", default="runs.csv", help="append one row per run")
    args = parser.parse_args()
 
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device={device} dataset={args.dataset} seed={args.seed} "
          f"adv={'on('+args.adv_attack+')' if args.adv_training else 'off'} "
          f"smooth={'on' if args.smooth_training else 'off'} eval_attack={args.eval_attack}")
 
 
    train_loader, test_loader, channels, num_classes, size = get_data(
        args.dataset,
        args.data_dir,
        args.batch_size,
    )

    model = SmallCNN(channels, num_classes,size).to(device) 
    cfg = SimpleNamespace(**vars(args))

    print("\n===Training===")
    train_single(cfg, model, train_loader, device)
    
    eval_attack = None
    if cfg.eval_attack != "none":
        eval_attack = make_attack(cfg.eval_attack, model, cfg.epsilon,
                                  cfg.attack_steps, cfg.step_size, cfg.sigma, cfg.eot_samples)
    
    print("\n=== Accuracy + per-class confidence ===")
    acc, per_class_conf = accuracy_with_class_conf(
        model, test_loader, device, attack=eval_attack, num_classes=num_classes)
    mean_conf = sum(per_class_conf) / len(per_class_conf)
    print(f"{'accuracy':24s}: {acc:.4f}  (eval_attack={cfg.eval_attack})")
    print(f"{'mean true-label conf':24s}: {mean_conf:.4f}")
    print(f"{'per-class confidence':24s}: " +
          ", ".join(f"{c}:{v:.3f}" for c, v in enumerate(per_class_conf)))
 
    record = {
        "seed": cfg.seed, "dataset": cfg.dataset, "epochs": cfg.epochs,
        "randomized_smoothing": str(cfg.smooth_training).lower(),
        "pgd_l2_defense": str(cfg.adv_training).lower(),
        "adv_attack_type": (cfg.adv_attack if cfg.adv_training else "none"),
        "eval_attack": cfg.eval_attack,
        "epsilon": cfg.epsilon, "sigma": cfg.sigma,
        "accuracy": acc, "mean_confidence": mean_conf,
    }
    for c, v in enumerate(per_class_conf):
        record[f"conf_class_{c}"] = v
 
    # --- certification: only for a smoothing-trained model ---
    if cfg.smooth_training:
        print("\n=== Randomized smoothing certification (clean inputs) ===")
        cert = certify_subset(model, test_loader, cfg, device, num_classes)
        rows = cert.pop("rows")
        for key, value in cert.items():
            print(f"{key:28s}: {value:.3f}" if isinstance(value, float) else f"{key:28s}: {value}")
        save_certification_rows(cfg.cert_csv, rows)
        record["cert_smoothed_accuracy"] = cert.get("smoothed_accuracy")
        record["cert_abstention_rate"] = cert.get("abstention_rate")
        record["cert_mean_radius"] = cert.get("mean_radius")
        record["cert_radius"] = cfg.radius
        record["cert_certified_accuracy"] = cert.get(f"certified_accuracy_at_{cfg.radius:g}")
 
    append_run_csv(cfg.results_csv, record, num_classes)
    print(f"\nappended run to {cfg.results_csv}")
 
 
if __name__ == "__main__":
    main()