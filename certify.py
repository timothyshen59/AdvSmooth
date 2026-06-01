import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint


ABSTAIN = -1

@torch.no_grad() 
def _noise_counts(model, x, n, sigma, num_classes, batch_size): 
    counts = torch.zeros(num_classes, dtype=torch.long, device=x.device) 
    remaining = n 
    
    while remaining > 0: 
        this_batch_size = min(batch_size, remaining)
        remaining -= this_batch_size
        
        batch = x.repeat((this_batch_size, 1, 1, 1))
        noise = torch.randn_like(batch) * sigma 
        
        predictions = model(batch+noise).argmax(1) 
        counts += torch.bincount(predictions, minlength=num_classes)
    
    return counts 
        
        
def _certify(model, x, sigma, n0, n, alpha, num_classes, batch_size): 
    model.eval() 
    
    c0 = _noise_counts(model, x, n0, sigma, num_classes, batch_size) 
    
    c_hat = c0.argmax().item() 
    
    c = _noise_counts(model, x, n, sigma,num_classes, batch_size) 
    
    nA =c[c_hat].item() 
    pA_bar = proportion_confint(nA, n, alpha = 2 * alpha, method = "beta")[0]
    
    if pA_bar < 0.5: 
        return ABSTAIN, 0.0
    
    return c_hat, float(sigma*norm.ppf(pA_bar))
    
    