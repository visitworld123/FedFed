import torchvision
from utils.log_info import *


# def train_reconst_images(args, client_index, data, mode,step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', 'client_{client_index}_train_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode), \
#              grid_X,step, tool=args.record_tool)
#
# def generate_reconst_images(args, client_index, data, mode, step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', "client_{client_index}_generate_{mode}.jpg".format(client_index=client_index, mode=mode), \
#              grid_X, tool=args.record_tool)
#
# def generate_reconst_images(args, client_index, data, mode, step, size=64):
#     grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
#     log_info('image', "client_{client_index}_test_{mode}.jpg".format(client_index=client_index, mode=mode), \
#              grid_X, step, tool=args.record_tool)
#


def train_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
    log_info('image', 'client_{client_index}_train_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)


def generate_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    grid_X = torchvision.utils.make_grid(data[:16], nrow=8, padding=2, normalize=True)
    log_info('image', 'client_{client_index}_generate_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)

def test_reconst_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    grid_X = torchvision.utils.make_grid(data[:64], nrow=8, padding=2, normalize=True)
    log_info('image', 'client_{client_index}_test_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)


def Share_constuct_images(client_index, data, mode, step, record_tool, wandb_record=False, size=64):
    grid_X = torchvision.utils.make_grid(data, nrow=8, padding=2, normalize=True)
    log_info('image', 'client_{client_index}_global_Batch_{mode}.jpg'.format(client_index=client_index, mode=mode),
             grid_X,step=step,record_tool=record_tool,
            wandb_record=wandb_record)
