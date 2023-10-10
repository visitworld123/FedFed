import torch
from torch.optim import Optimizer

from .base_lr_scheduler import _LRScheduler



class MultiStepLR(_LRScheduler):
    
    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_decay_rate=0.1, lr_milestones=[30, 60, 90]):
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )

        self.lr_decay_rate = lr_decay_rate
        self.lr_milestones = lr_milestones



    def get_lr(self, progress):
        index = 0
        for milestone in self.lr_milestones:
            if progress < milestone:
                break
            else:
                index += 1
        self.lr = self.base_lr * (self.lr_decay_rate**index)
        return self.lr









