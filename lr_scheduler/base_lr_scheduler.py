import logging
from abc import ABC, abstractmethod

import torch
from torch.optim import Optimizer



class _LRScheduler(object):

    def __init__(self, optimizer, base_lr, warmup_epochs=0, num_iterations=0,
                lr_warmup_type="constant", lr_warmup_value=0.1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.num_iterations = num_iterations
        self.lr_warmup_type = lr_warmup_type
        self.lr_warmup_value = lr_warmup_value

    def update_groups(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def step(self, progress):
        if progress < self.warmup_epochs:
            return "warmup"
        else:
            lr = self.get_lr(progress)
            self.update_groups(lr)
            return "step"

    def warmup_step(self, iterations):
        if self.lr_warmup_type == "constant":
            self.lr = self.lr_warmup_value
        elif self.lr_warmup_type == "gradual":
            warmup_total_iters = self.num_iterations * self.warmup_epochs
            min_lr = self.base_lr / warmup_total_iters 
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            self.lr = min_lr + lr_interval * iterations
        else:
            raise NotImplementedError
        self.update_groups(self.lr)
        # logging.info(f"Scheduler::: !!!!  iterations: {iterations} self.lr: {self.lr}\n\n")


    @abstractmethod
    def get_lr(self, progress):
        """ define this function for step() using.
        """
        pass







