import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import random
import numpy as np


from model import SmallCNN
from defenses import train_gaussian 
from certify import ABSTAIN, _certify
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)      
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_data(dataset, batch_size = 128): 
    mnist_transform = transforms.ToTensor()
    
    
    cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
    ])
    
    print(f"[dataset] Using {dataset}")
    if dataset == "mnist": 
        training_split = datasets.MNIST("./data", train=True, download=True, transform=mnist_transform)
        testing_split = datasets.MNIST("./data", train=False, download=True, transform=mnist_transform)
        channels, num_classes, size = 1, 10, 28 

    else: 
        training_split = datasets.CIFAR10("./data", train=True, download=True, transform=cifar_transform)
        testing_split = datasets.CIFAR10("./data", train=False, download=True, transform=cifar_transform)
        channels, num_classes, size = 3, 10, 32   
        
    return (DataLoader(training_split, batch_size, shuffle=True, num_workers=2),
            DataLoader(testing_split, batch_size, shuffle=False, num_workers=2),
            channels, num_classes, size)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    #Seeding 
    ap.add_argument("--seed", type=int, default=42)  
    #Training 
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.25)
    # Certification
    ap.add_argument("--radius", type=float, default=0.5, help="report certified accuracy at this L2 radius")
    ap.add_argument("--n_certify", type=int, default=500, help="test images to certify")
    ap.add_argument("--n0", type=int, default=100, help="samples for class selection")
    ap.add_argument("--n", type=int, default=1000, help="samples for estimation (~100000 for tight certs)")
    ap.add_argument("--alpha", type=float, default=0.001, help="certification failure probability")
    ap.add_argument("--certify_batch", type=int, default=200)
    args = ap.parse_args()
    
    set_seed(args.seed)
    train_loader, test_loader, ch, ncls, size = get_data(args.dataset)
    model = SmallCNN(ch, ncls, size).to(DEVICE)
 

    print(f"device {DEVICE} | training base classifier f (sigma={args.sigma})")
    train_gaussian(args, model, train_loader, DEVICE)
 
    # grab a test subset (test loader is unshuffled, so order is stable)
    xs, ys, c = [], [], 0
    for x, y in test_loader:
        xs.append(x); ys.append(y); c += x.size(0)
        if c >= args.n_certify:
            break
    X = torch.cat(xs)[:args.n_certify]
    Y = torch.cat(ys)[:args.n_certify]
 
    correct = certified = abstains = 0
    radius_sum = 0.0
    for i in range(X.size(0)):
        x_i = X[i].unsqueeze(0).to(DEVICE)          # (C,H,W) -> (1,C,H,W) for _noise_counts' repeat
        pred, r = _certify(model, x_i, args.sigma, args.n0, args.n,
                           args.alpha, ncls, args.certify_batch)
        radius_sum += r
        if pred == ABSTAIN:
            abstains += 1
        if pred == int(Y[i]):
            correct += 1                            # smoothed prediction correct
            if r >= args.radius:
                certified += 1                      # ...and provably so within `radius`
 
    N = X.size(0)
    print("\n=== randomized smoothing evaluation ===")
    print(f"images certified            : {N}")
    print(f"smoothed prediction accuracy: {correct / N:.3f}")
    print(f"abstention rate             : {abstains / N:.3f}")
    print(f"mean certified L2 radius    : {radius_sum / N:.3f}")
    print(f"certified accuracy @ r>={args.radius}: {certified / N:.3f}")
 
 
if __name__ == "__main__":
    main()
 