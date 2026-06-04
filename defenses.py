import torch 
import torch.nn.functional as F 
from tqdm import tqdm

from attacks import L2PGDAttack, EOTPGDL2

def _optim(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)
 
 
def _attack_kwargs(config, train_attack_config):
    if train_attack_config is not None:
        if hasattr(train_attack_config, "items"):
            return dict(train_attack_config)
        return vars(train_attack_config)

    return {
        "epsilon": config.train_epsilon,
        "k": config.train_attack_steps,
        "a": config.train_step_size,
        "random_start": True,
        "loss_func": "xent",
    }


def train_adversarial(config, model, loader, device, train_attack_config=None):
    optimizer = _optim(config, model) 
    attack_kwargs = _attack_kwargs(config, train_attack_config)
    
    if config.adv_attack == "pgd_l2":
        print("\n===== PGD L2 Attack======")
        attack = L2PGDAttack(model, **attack_kwargs)

    elif config.adv_attack == "eot":
        print("\n===== EOT PGD L2 Attack======")
        attack = EOTPGDL2(model, sigma = config.sigma, m=config.eot_samples,**attack_kwargs)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total = 0

    
        for x, y in tqdm(loader, desc=f"l2 defense train {epoch + 1}/{config.epochs}"):
            x, y = x.to(device), y.to(device)
            x_adv = attack.perturb(x, y)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total += y.numel()
        print(f"[l2_defense] epoch={epoch + 1}/{config.epochs} loss={total_loss / total:.4f}")


def train_standard(config, model, loader, device): 
    """
    Train Base Classifier with no defensive mechanisms 
    """
    
    optimizer = _optim(config, model) 
    model.train() 
    
    for epoch in range(config.epochs): 
        for x,y in loader: 
            x,y = x.to(device), y.to(device) 
            optimizer.zero_grad()
            
            loss = F.cross_entropy(model(x), y)
            loss.backward() 
            optimizer.step() 
        
        print(f"[standard] Epoch: {epoch+1}/{config.epochs}| Loss: {loss.item():.3f}")


def train_gaussian(config, model, loader, device): 
    """
    Train Base Classifier for randomized smoothing under Gaussian noise-added samples 
    """
    optimizer = _optim(config, model)
    model.train()
    
    for epoch in range(config.epochs): 
        for x,y in loader: 
            x,y = x.to(device), y.to(device) 
            # applies pertubation to the input
            x_noisy = (x + torch.randn_like(x) * config.sigma) .clamp(0, 1)
            optimizer.zero_grad()
            
            loss = F.cross_entropy(model(x_noisy), y) 
            loss.backward()
            optimizer.step() 
            
        print(f"[rand_smooth] Epoch: {epoch+1}/{config.epochs}| Loss: {loss.item():.3f}")
        
