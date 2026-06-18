"""
L2 adversarial attacks for robustness evaluation and adversarial training.

Provides two PGD-based attacks in the L2 norm:
    - L2PGDAttack: Standard PGD attack (Madry et al., 2018).
    - EOTPGDL2: EOT-PGD attack for attacking smoothed classifiers by
      averaging gradients over Gaussian noise samples (Athalye et al., 2018).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class L2PGDAttack:
    """
    Projected Gradient Descent attack in the L2 norm (Madry et al., 2018).

    Iteratively perturbs inputs by following the gradient of the loss,
    projecting back onto the L2 ball of radius epsilon after each step.

    Supports cross-entropy and Carlini-Wagner loss functions.
    """

    def __init__(
        self,
        model: Module,
        epsilon: float = 0.5,
        k: int = 20,
        a: float = 0.1,
        random_start: bool = True,
        loss_func: str = "xent",
    ) -> None:
        """
        Args:
            model: The classifier to attack.
            epsilon: L2 perturbation budget.
            k: Number of PGD iterations.
            a: Step size per iteration.
            random_start: If True, initialise from a random point within
                          the L2 ball before iterating.
            loss_func: Loss function to maximise, either 'xent' (cross-entropy)
                       or 'cw' (Carlini-Wagner).
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.random_start = random_start
        self.loss_func = loss_func

    def _loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """
        Compute the attack loss from logits and true labels.

        Args:
            logits: Raw model outputs of shape (B, num_classes).
            y: True labels of shape (B,).

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If loss_func is not 'xent' or 'cw'.
        """
        if self.loss_func == "xent":
            return F.cross_entropy(logits, y)
        if self.loss_func == "cw":
            one_hot = F.one_hot(y, num_classes=10).float()
            correct_logit = torch.sum(one_hot * logits, dim=1)
            wrong_logit = torch.max(
                (1.0 - one_hot) * logits - 1e4 * one_hot, dim=1
            ).values
            return -F.relu(correct_logit - wrong_logit + 50.0).mean()
        raise ValueError("loss_func must be 'xent' or 'cw'")

    def _project_l2(self, x_adv: Tensor, x_nat: Tensor) -> Tensor:
        """
        Project x_adv onto the L2 ball of radius epsilon centred at x_nat,
        then clip to the valid pixel range [0, 1].

        Args:
            x_adv: Perturbed input of shape (B, C, H, W).
            x_nat: Original clean input of shape (B, C, H, W).

        Returns:
            Projected input of shape (B, C, H, W).
        """
        delta = x_adv - x_nat
        flat = delta.view(delta.shape[0], -1)
        norm = flat.norm(p=2, dim=1).clamp(min=1e-12)
        factor = torch.minimum(torch.ones_like(norm), self.epsilon / norm)
        delta = delta * factor.view(-1, 1, 1, 1)
        return torch.clamp(x_nat + delta, 0.0, 1.0)

    def perturb(self, x_nat: Tensor, y: Tensor) -> Tensor:
        """
        Generate adversarial examples for a batch of inputs.

        Args:
            x_nat: Clean input tensor of shape (B, C, H, W), values in [0, 1].
            y: True labels of shape (B,).

        Returns:
            Adversarial examples of shape (B, C, H, W), detached from the
            computation graph.
        """
        self.model.eval()

        if self.random_start:
            # Adds randomised noise that stays within epsilon ball
            delta = torch.randn_like(x_nat)  # random direction vector

            flat = delta.view(
                delta.shape[0], -1
            )  # flatten to (B, C*H*W) for norm computation
            norm = flat.norm(p=2, dim=1).clamp(min=1e-12)  # normalise to unit L2 sphere

            radius = torch.rand(
                x_nat.shape[0], device=x_nat.device
            )  # Initialize random radius to start at
            delta = delta / norm.view(-1, 1, 1, 1)  # Normalise and scale
            delta = delta * (radius * self.epsilon).view(-1, 1, 1, 1)

            x = torch.clamp(x_nat + delta, 0.0, 1.0)  # Clamps back to 0-1 pixel range
        else:
            x = x_nat.clone()

        for _ in range(self.k):
            # enable grad tracking for current x
            x.requires_grad_(True)
            logits = self.model(x)
            loss = self._loss(logits, y)
            grad = torch.autograd.grad(loss, x)[0]

            # normalise gradient to unit L2 norm for a fixed step size
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).clamp(min=1e-12)
            x = x.detach() + self.a * grad.detach() / grad_norm.view(-1, 1, 1, 1)

            # project back onto L2 ball and clip to valid pixel range
            x = self._project_l2(x, x_nat)

        return x.detach()


class EOTPGDL2(L2PGDAttack):
    """
    Expectation over Transformation PGD attack in the L2 norm (Athalye et al., 2018).

    Extends PGD-L2 attack to smoothed classifiers by averaging gradients
    over m noisy samples at each iteration, matching the Gaussian noise
    convention used during randomized smoothing training and certification.
    """

    def __init__(self, model: Module, sigma: float, m: int = 8, **kw) -> None:
        """
        Args:
            model: The classifier to attack.
            sigma: Standard deviation of Gaussian noise, must match the
                   sigma used during smoothing training and certification.
            m: Number of noise samples to average gradients over per step.
            **kw: Additional keyword arguments passed to L2PGDAttack.
        """
        super().__init__(model, **kw)
        self.sigma = sigma
        self.m = m

    def perturb(self, x_nat: Tensor, y: Tensor) -> Tensor:
        """
        Generate adversarial examples against the smoothed classifier.

        At each PGD step, estimates the gradient by averaging over m forward
        passes with independent Gaussian noise, then takes a normalised
        gradient step and projects back onto the L2 ball.

        Args:
            x_nat: Clean input tensor of shape (B, C, H, W), values in [0, 1].
            y: True labels of shape (B,).

        Returns:
            Adversarial examples of shape (B, C, H, W), detached from the
            computation graph.
        """
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
                noise = (
                    torch.randn_like(xn) * self.sigma
                )  # Gaussian noise for smoothign
                loss = self._loss(self.model(xn + noise), y)
                grad = grad + torch.autograd.grad(loss, xn)[0]

            grad = grad / self.m  # Average gradient over m samples
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).clamp(min=1e-12)
            x = x.detach() + self.a * grad / grad_norm.view(-1, 1, 1, 1)
            x = self._project_l2(x, x_nat)

        return x.detach()
