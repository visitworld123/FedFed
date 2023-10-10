import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from .normal_trainer import NormalTrainer

from optim.build import create_optimizer
from loss_fn.build import create_loss
from lr_scheduler.build import create_scheduler





def create_trainer(args, device, model=None, **kwargs):

    params = None
    optimizer = create_optimizer(args, model, params=params, **kwargs)


    criterion = create_loss(args, device, **kwargs)
    lr_scheduler = create_scheduler(args, optimizer, **kwargs)   # no for FedAvg

    model_trainer = NormalTrainer(model, device, criterion, optimizer, lr_scheduler, args, **kwargs)

    return model_trainer











