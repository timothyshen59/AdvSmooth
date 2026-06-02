import csv

import torch
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


def _certify_limit(cfg):
    limit = getattr(cfg, "certify_limit", None)
    if limit is None:
        limit = getattr(cfg, "n_certify", None)
    if limit is None:
        raise AttributeError("certification config must define certify_limit or n_certify")
    return limit


def certify_subset(model, loader, cfg, device, num_classes):
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
    
    
