
# from .config import CfgNode as CN
from .config import CfgNode as CN

_C = CN()

_C.dataset = 'cifar10'
_C.client_num_in_total = 10
_C.client_num_per_round = 5
_C.gpu_index = 0 # for centralized training or standalone usage
_C.num_classes = 10
_C.data_dir = './../data'
_C.partition_method = 'hetero'
_C.partition_alpha = 0.1
_C.model = 'resnet18_v2'
_C.model_input_channels = 3
_C.model_output_dim = 10
_C.algorithm = 'FedAvg'
# fedprox
_C.fedprox = False
_C.fedprox_mu = 0.1
_C.scaffold = False

_C.global_epochs_per_round = 1
_C.comm_round = 1000

_C.lr = 0.01
_C.seed = 2

_C.record_tool = 'wandb'  # using wandb or tensorboard
_C.wandb_record = False

_C.batch_size = 64
_C.VAE_batch_size = 64
_C.VAE_aug_batch_size = 64

_C.VAE_re = 5.0
_C.VAE_ce = 2.0
_C.VAE_kl = 0.005

_C.VAE_std1 = 0.2
_C.VAE_std2 = 0.25
_C.VAE_x_ce = 0.4

_C.VAE_comm_round = 15
_C.VAE_client_num_per_round = 10
_C.VAE_adaptive = True
_C.noise_type = 'Gaussian'  # Gaussian or Laplace


# ---------------------------------------------------------------------------- #
# mode settings
# ---------------------------------------------------------------------------- #
_C.mode = 'standalone'  # standalone or centralized
_C.test = True
_C.instantiate_all = True

_C.client_index = 0

# ---------------------------------------------------------------------------- #
# task settings
# ---------------------------------------------------------------------------- #
_C.task = 'classification' #    ["classification", "stackoverflow_lr", "ptb"]


# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #

_C.dataset_aug = "default"
_C.dataset_resize = False
_C.dataset_load_image_size = 32

_C.data_efficient_load = True    #  Efficiently load dataset, only load one full dataset, but split to many small ones.

_C.dirichlet_min_p = None #  0.001    set dirichlet min value for letting each client has samples of each label
_C.dirichlet_balance = False # This will try to balance dataset partition among all clients to make them have similar data amount

_C.data_load_num_workers = 1

# ---------------------------------------------------------------------------- #
# data sampler
# ---------------------------------------------------------------------------- #
_C.data_sampler = "random"  # 'random'

_C.TwoCropTransform = False


# ---------------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------------- #

_C.model_out_feature = False
_C.model_out_feature_layer = "last"
_C.model_feature_dim = 512

_C.pretrained = False
_C.pretrained_dir = " "



# ---------------------------------------------------------------------------- #
# generator
# ---------------------------------------------------------------------------- #
_C.image_resolution = 32

# ---------------------------------------------------------------------------- #
# Client Select
# ---------------------------------------------------------------------------- #
_C.client_select = "random"  #   ood_score, ood_score_oracle


# ---------------------------------------------------------------------------- #
# loss function
# ---------------------------------------------------------------------------- #
_C.loss_fn = 'CrossEntropy'
_C.exchange_model = True



# ---------------------------------------------------------------------------- #
# optimizer settings
# comm_round is only used in FedAvg.
# ---------------------------------------------------------------------------- #
_C.max_epochs = 90

_C.client_optimizer = 'no' # Please indicate which optimizer is used, if no, set it as 'no'
_C.server_optimizer = 'no'


_C.wd = 0.0001
_C.momentum = 0.9
_C.nesterov = False


# ---------------------------------------------------------------------------- #
# Learning rate schedule parameters
# ---------------------------------------------------------------------------- #
_C.sched = 'no'   # no (no scheudler), StepLR MultiStepLR  CosineAnnealingLR
_C.lr_decay_rate = 0.992
_C.step_size = 1
_C.lr_milestones = [30, 60]
_C.lr_T_max = 10
_C.lr_eta_min = 0
_C.lr_warmup_type = 'constant' # constant, gradual.
_C.warmup_epochs = 0
_C.lr_warmup_value = 0.1


# ---------------------------------------------------------------------------- #
# logging
# ---------------------------------------------------------------------------- #
_C.level = 'INFO' # 'INFO' or 'DEBUG'


# ---------------------------------------------------------------------------- #
# VAE settings
# ---------------------------------------------------------------------------- #
_C.VAE = True
_C.VAE_local_epoch = 1
_C.VAE_d = 32
_C.VAE_z = 2048
_C.VAE_sched = 'cosine'
_C.VAE_sched_lr_ate_min = 2.e-3
_C.VAE_step = '+'

_C.VAE_mixupdata = False
_C.VAE_curriculum = True # Curriculum for reconstruction term which helps for better convergence

_C.VAE_mean = 0

_C.VAE_alpha = 2.0
_C.VAE_curriculum = True
