import torch


"""
    args.lr_scheduler in 
    ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    --step-size
    --lr-decay-rate
    --lr-milestones
    --lr-T-max
    --lr-eta-min
"""

def create_scheduler(args, optimizer):
    if args.sched == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.step_size, args.lr_decay_rate)
    elif args.sched == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.lr_milestones, args.lr_decay_rate)
    elif args.sched == "ExponentialLR":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, args.lr_decay_rate)
    elif args.sched == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.lr_T_max, args.lr_eta_min)
    elif args.sched == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_sheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10,
            verbose=False, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-08)
    else:
        raise NotImplementedError

    return lr_scheduler







