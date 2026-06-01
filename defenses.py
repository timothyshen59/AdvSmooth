import torch 
import torch.nn.functional as F 
from tqdm import tqdm

from attacks import L2PGDAttack

def _optim(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)
 
 

def train_adversarial(config, model, loader, epochs, device, train_attack_config):
    optimizer = _optim(config, model) 


    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total = 0
        attack = L2PGDAttack(model, train_attack_config)

        for x, y in tqdm(loader, desc=f"defense train epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            x_adv = attack.perturb(x, y)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total += y.numel()
        print(f"defense epoch={epoch + 1} loss={total_loss / total:.4f}")


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
            x_noisy = (x + torch.randn_like(x) * config.sigma) #TODO: Check rescaling and pixel value range 
            optimizer.zero_grad()
            
            loss = F.cross_entropy(model(x_noisy), y) 
            loss.backward()
            optimizer.step() 
            
        print(f"[rand_smooth] Epoch: {epoch+1}/{config.epochs}| Loss: {loss.item():.3f}")
        


