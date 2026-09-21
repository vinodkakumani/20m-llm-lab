"""Optimizer construction for the baseline training loop."""
import torch
from torch import nn

def build_adamw(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=learning_rate, betas=(0.9, 0.95))
