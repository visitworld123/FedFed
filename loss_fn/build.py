import torch.nn as nn



def create_loss(args, device=None, **kwargs):
    if "client_index" in kwargs:
        client_index = kwargs["client_index"]
    else:
        client_index = args.client_index

    if args.loss_fn == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "nll_loss":
        loss_fn = nn.NLLLoss()
    else:
        raise NotImplementedError

    return loss_fn















