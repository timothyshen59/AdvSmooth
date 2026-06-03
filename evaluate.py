import argparse
import torch
import torch.nn.functional as F 
from tqdm import tqdm


from model import SmallCNN
from defenses import train_gaussian 
from certify import certify_subset
from utils import get_data, get_device, set_seed


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
 


def accuracy_with_class_conf(model, loader, device, attack=None, num_classes=10):
    model.eval()
    correct = 0
    total = 0
    
    confidence_sum = torch.zeros(num_classes)
    class_cnt = torch.zeros(num_classes)
    
    for x, y in tqdm(loader, desc="accuracy", leave=False):
        x, y = x.to(device), y.to(device)
        if attack is not None:
            x = attack.perturb(x, y)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(1)
            true_prob = probs.gather(1, y.view(-1,1)).squeeze(1)
            
        correct += (pred == y).sum().item()
        total += y.numel()
        
        #Count counfidence score 
        for c in range(num_classes): 
            mask = (y==c)
            if mask.any(): 
                confidence_sum[c] += true_prob[mask].sum().item()
                class_cnt[c] += mask.sum().item()
    
    per_class_confidence = (confidence_sum / class_cnt.clamp(min=1)).tolist()
    return correct / total, per_class_confidence



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    ap.add_argument("--device", default="auto")
    #Seeding 
    ap.add_argument("--seed", type=int, default=42)  
    #Training 
    ap.add_argument("--epochs", type=int, default=2)
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
    device = get_device(args.device)
    train_loader, test_loader, ch, ncls, size = get_data(args.dataset, batch_size=128, num_workers=2)
    model = SmallCNN(ch, ncls, size).to(device)
 

    print(f"device {device} | training base classifier f (sigma={args.sigma})")
    train_gaussian(args, model, train_loader, device)

    cert = certify_subset(model, test_loader, args, device, ncls)
    cert.pop("rows")

    print("\n=== randomized smoothing evaluation ===")
    for key, value in cert.items():
        print(f"{key:28s}: {value:.3f}" if isinstance(value, float) else f"{key:28s}: {value}")
 
 
if __name__ == "__main__":
    main()
 
