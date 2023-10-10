import torch

from .steplr_scheduler import StepLR
from .multisteplr_scheduler import MultiStepLR
from .consine_lr_scheduler import CosineAnnealingLR

"""
    args.lr_scheduler in 
    ["StepLR", "MultiStepLR", "CosineAnnealingLR"]
    --step-size
    --lr-decay-rate
    --lr-milestones
    --lr-T-max
    --lr-eta-min
"""


def create_scheduler(args, optimizer, **kwargs):
    """
        num_iterations is the number of iterations per epoch.
    """
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index
    if args.sched == "no":
        lr_scheduler = None
    elif args.sched == "StepLR":
        lr_scheduler = StepLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_decay_rate=args.lr_decay_rate,
            step_size=args.step_size)
    elif args.sched == "MultiStepLR":
        lr_scheduler = MultiStepLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_decay_rate=args.lr_decay_rate,
            lr_milestones=args.lr_milestones)
    elif args.sched == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(
            optimizer, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
            num_iterations=kwargs['num_iterations'],
            lr_warmup_type=args.lr_warmup_type, lr_warmup_value=args.lr_warmup_value,
            lr_T_max=args.lr_T_max,
            lr_eta_min=args.lr_eta_min)
    else:
        raise NotImplementedError

    return lr_scheduler







