import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

class SmallCNN(nn.Module): 
    def __init__(self, input_channel, num_classes, size):
        super().__init__()
        conv_layers = [
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
        ]
        
        dummy_input = torch.zeros(1, input_channel, size, size) 
        with torch.no_grad(): 
            flattened_dim = nn.Sequential(*conv_layers)(dummy_input).flatten(1).shape[1]
        
        
        self.net = nn.Sequential(
            *conv_layers, 
            nn.Flatten(), 
            nn.Linear(flattened_dim, 100), 
            nn.ReLU(), 
            nn.Linear(100, num_classes), 
        )
    
    def forward(self, x): 
        return self.net(x) 
    