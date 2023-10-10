
import torch
from torch.optim import Optimizer

from .base_lr_scheduler import _LRScheduler


class StepLR(_LRScheduler):

    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1,
                lr_decay_rate=0.97, step_size=1):
        super().__init__(
            optimizer, base_lr, warmup_epochs, num_iterations,
            lr_warmup_type, lr_warmup_value
        )

        self.lr_decay_rate = lr_decay_rate
        self.step_size = step_size


    def get_lr(self, progress):
        # This aims to make a float step_size work.
        exp_num = progress / self.step_size
        self.lr = self.base_lr * (self.lr_decay_rate**exp_num)
        return self.lr











