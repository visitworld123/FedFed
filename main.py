import argparse
import logging
import os
import socket
import sys

import numpy as np
import torch
# add the FedML root directory to the python path

from utils.logger import logging_config
from configs import get_cfg

from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
from algorithms_standalone.fednova.FedNovaManager import FedNovaManager

from utils.set import *

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument("--config_file", default=None, type=str)
    parser.add_argument("opts", help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    # parse python script input parameters

    #----------loading personalized params-----------------#
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args.config_file)
    #### set up cfg ####
    # default cfg
    cfg = get_cfg()

    cfg.setup(args)

    # Build config once again
    #cfg.setup(args)
    cfg.mode = 'standalone'

    cfg.server_index = -1
    cfg.client_index = -1
    seed = cfg.seed
    process_id = 0
    # show ultimate config
    logging.info(dict(cfg))

    #-------------------customize the process name-------------------
    str_process_name = cfg.algorithm + " (standalone):" + str(process_id)

    logging_config(args=cfg, process_id=process_id)

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                ", host name = " + hostname + "########" +
                ", process ID = " + str(os.getpid()))

    set_random_seed(seed) 
    torch.backends.cudnn.deterministic =True

    device = torch.device("cuda:" + str(cfg.gpu_index) if torch.cuda.is_available() else "cpu")
    if cfg.record_tool == 'wandb' and cfg.wandb_record:
        import wandb
        wandb.init(config=args, name='test',
                   project='CIFAR10')
    else: 
        os.environ['WANDB_MODE'] = 'dryrun'

    if cfg.algorithm == 'FedAvg':
        fedavg_manager = FedAVGManager(device, cfg)
        fedavg_manager.train()
    elif cfg.algorithm == 'FedNova':
        fednova_manager = FedNovaManager(device, cfg)
        fednova_manager.train()
    else:
        raise NotImplementedError








