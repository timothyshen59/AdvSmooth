# Adversarial Smoothing for L2 Robustness

A PyTorch implementation of randomized smoothing for certifying L2 robustness 
of neural networks, with support for adversarial training and EOT-PGD evaluation.

## Overview

This project implements and evaluates certified robustness methods for image 
classifiers under L2 adversarial perturbations. It supports three training 
regimes — standard, adversarial (PGD-L2), and Gaussian noise augmentation for 
randomized smoothing — and evaluates them under clean and adversarial conditions.
Certified accuracy and robustness radii are computed using the Cohen et al. (2019) 
certification procedure.

## Report

A full write-up of the experiments and results is available here: [Report](report.pdf)

## Project Structure

```
project/
├── main.py              # Entry point: training, evaluation, certification
├── core/
│   ├── model.py         # SmallCNN base classifier
│   └── utils.py         # Seeding, device selection, data loading
├── attacks/
│   └── attacks.py       # L2PGDAttack, EOTPGDL2
├── defenses/
│   └── defenses.py      # Standard, Gaussian, and adversarial training
├── eval/
│   ├── evaluate.py      # Accuracy and per-class confidence
│   └── certify.py       # Randomized smoothing certification
└── results/
    ├── csv_utils.py     # CSV logging
    └── visualize_results.py     # Result plotting
```

## Requirements

Python 3.9+ and PyTorch 2.0+ recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quickstart


```bash
# Install dependencies
pip install -r requirements.txt


# Standard training and evaluation
python main.py --dataset mnist

# Randomized smoothing training and certification
python main.py --dataset mnist --smooth-training --eval-attack eot

# Adversarial training
python main.py --dataset mnist --adv-training --adv-attack pgd_l2

# Load a saved model and evaluate
python main.py --load-model --run-name my_run --eval-attack eot

# Generate plots from results
python results/visualize_results.py --csv runs.csv --dataset mnist --epsilon 0.5
```

## References

[1] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu,
"Towards deep learning models resistant to adversarial attacks,"
in *International Conference on Learning Representations (ICLR)*, 2018.
https://arxiv.org/abs/1706.06083

[2] J. Cohen, E. Rosenfeld, and Z. Kolter,
"Certified adversarial robustness via randomized smoothing,"
in *International Conference on Machine Learning (ICML)*, 2019.
https://arxiv.org/abs/1902.02918

[3] H. Salman, G. Yang, J. Li, P. Zhang, H. Zhang, I. Razenshteyn, and S. Bubeck,
"Provably robust deep learning via adversarially trained smoothed classifiers,"
in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
https://arxiv.org/abs/1906.04584


