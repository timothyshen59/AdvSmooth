import torch 
import torch.nn.functional as F 

class L2PGDAttack:
    def __init__(self, model, epsilon=0.5, k=20, a=0.1, random_start=True, loss_func="xent"):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.random_start = random_start
        self.loss_func = loss_func

    def _loss(self, logits, y):
        if self.loss_func == "xent":
            return F.cross_entropy(logits, y)
        if self.loss_func == "cw":
            one_hot = F.one_hot(y, num_classes=10).float()
            correct_logit = torch.sum(one_hot * logits, dim=1)
            wrong_logit = torch.max((1.0 - one_hot) * logits - 1e4 * one_hot, dim=1).values
            return -F.relu(correct_logit - wrong_logit + 50.0).mean()
        raise ValueError("loss_func must be 'xent' or 'cw'")

    def _project_l2(self, x_adv, x_nat):
        delta = x_adv - x_nat
        flat = delta.view(delta.shape[0], -1)
        norm = flat.norm(p=2, dim=1).clamp(min=1e-12)
        factor = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
        delta = delta * factor.view(-1, 1, 1, 1)
        return torch.clamp(x_nat + delta, 0.0, 1.0)

    def perturb(self, x_nat, y):
        self.model.eval()

        if self.random_start:
            delta = torch.randn_like(x_nat)
            flat = delta.view(delta.shape[0], -1)
            norm = flat.norm(p=2, dim=1).clamp(min=1e-12)
            radius = torch.rand(x_nat.shape[0], device=x_nat.device)
            delta = delta / norm.view(-1, 1, 1, 1)
            delta = delta * (radius * self.epsilon).view(-1, 1, 1, 1)
            x = torch.clamp(x_nat + delta, 0.0, 1.0)
        else:
            x = x_nat.clone()

        for _ in range(self.k):
            x.requires_grad_(True)
            logits = self.model(x)
            loss = self._loss(logits, y)
            grad = torch.autograd.grad(loss, x)[0]
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).clamp(min=1e-12)
            x = x.detach() + self.a * grad.detach() / grad_norm.view(-1, 1, 1, 1)
            x = self._project_l2(x, x_nat)

        return x.detach()
    
class EOTPGDL2(L2PGDAttack): 
    def __init__(self, model, sigma, m=8, **kw):
        super().__init__(model, **kw)
        self.sigma = sigma       
        self.m = m 
        
    def perturb(self, x_nat, y): 
        self.model.eval()

        if self.random_start:
            delta = torch.randn_like(x_nat)
            flat = delta.view(delta.shape[0], -1)
            norm = flat.norm(p=2, dim=1).clamp(min=1e-12)
            radius = torch.rand(x_nat.shape[0], device=x_nat.device)
            delta = delta / norm.view(-1, 1, 1, 1)
            delta = delta * (radius * self.epsilon).view(-1, 1, 1, 1)
            x = torch.clamp(x_nat + delta, 0.0, 1.0)
        else:
            x = x_nat.clone()
            
  
        for _ in range(self.k):
            
            grad = torch.zeros_like(x)

            for _ in range(self.m):
                xn = x.clone().detach().requires_grad_(True)
                noise = torch.randn_like(xn) * self.sigma     
                loss = self._loss(self.model(xn + noise), y)
                grad = grad + torch.autograd.grad(loss, xn)[0]
            
            grad = grad/self.m
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).clamp(min=1e-12)
            x = x.detach() + self.a * grad / grad_norm.view(-1, 1, 1, 1)
            x = self._project_l2(x, x_nat)
        
        return x.detach() 