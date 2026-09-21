"""Warmup then cosine learning-rate scheduling."""
import math, torch
def build_cosine_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps: raise ValueError("Require 0 <= warmup_steps < total_steps.")
    def factor(step: int) -> float:
        if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
