import torch
from torch.optim import Optimizer

import numpy as np

from .base_lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0, 
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_T_max=100, lr_eta_min=0):
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )

        self.lr_T_max = lr_T_max
        self.lr_eta_min = lr_eta_min

    def get_lr(self, progress):
        e = progress - self.warmup_epochs
        es = self.lr_T_max - self.warmup_epochs
        lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
        self.lr = lr
        return self.lr





