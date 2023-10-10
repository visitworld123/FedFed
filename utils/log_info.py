from torch.utils.tensorboard import SummaryWriter


def log_info(type:str, name:str, info, step=None, record_tool='wandb', wandb_record=False):
    '''
    type: the info type mainly include: image, scalar (tensorboard may include hist, scalars)
    name: replace the info name displayed in wandb or tensorboard
    info: info to record
    '''
    if record_tool == 'wandb':
        import wandb
    if type == 'image':
        if record_tool == 'tensorboard':
            writer.add_image(name, info, step)
        if record_tool == 'wandb' and wandb_record:
            wandb.log({name: wandb.Image(info)})

    if type == 'scalar':
        if record_tool == 'tensorboard':
            writer.add_scalar(name, info, step)
        if record_tool == 'wandb'and wandb_record:
            wandb.log({name:info})
    if type == 'histogram':
        writer.add_histogram(name, info, step)