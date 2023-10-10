from data_preprocessing.utils.stats import get_dataset_image_size
import logging

from model.cv.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet10

from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.FL_VAE import *


CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]


def create_model(args, model_name, output_dim, pretrained=False, device=None, **kwargs):
    model = None
    logging.info(f"model name: {model_name}")

    if model_name in RNN_MODEL_LIST:
        pass
    else:
        image_size = get_dataset_image_size(args.dataset)

    if model_name == "vgg-9":
        if args.dataset in ("mnist", 'femnist', 'fmnist'):
            model = ModerateCNNMNIST(output_dim=output_dim,
                                        input_channels=args.model_input_channels)
        elif args.dataset in ("cifar10", "cifar100", "cinic10", "svhn"):
            # print("in moderate cnn")
            model = ModerateCNN(args, output_dim=output_dim)
            print("------------------params number-----------------------")
            num_params = sum(param.numel() for param in model.parameters())
            print(num_params)
    elif model_name == "resnet18_v2":
        logging.info("ResNet18_v2")
        model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
    elif model_name == "resnet34_v2":
        logging.info("ResNet34_v2")
        model = ResNet34(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels, device=device)
    elif model_name == "resnet50_v2":
        model = ResNet50(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels)
    elif model_name == "resnet10_v2":
        logging.info("ResNet10_v2")
        model = ResNet10(args=args, num_classes=output_dim, image_size=image_size,
                            model_input_channels=args.model_input_channels, device=device)
    else:
        raise NotImplementedError

    return model
