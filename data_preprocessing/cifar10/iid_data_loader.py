import os
import argparse
import time
import math
import logging

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

from .transform import data_transforms_cifar10


def load_iid_cifar10(dataset, data_dir, partition_method, 
        partition_alpha, client_number, batch_size, rank=0, args=None):

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    # CIFAR_STD = [0.2023, 0.1994, 0.2010]

    image_size = 32
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN , std=CIFAR_STD),
        ])

    train_dataset = CIFAR10(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR10(root=data_dir, train=False,
                            transform=test_transform, download=False)

    if args.mode in ['distributed', 'centralized']:
        train_sampler = None
        shuffle = True
        if client_number > 1:
            train_sampler = data.distributed.DistributedSampler(
                train_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)
            shuffle = False

            # Note that test_sampler is for distributed testing to accelerate training
            test_sampler = data.distributed.DistributedSampler(
                test_dataset, num_replicas=client_number, rank=rank)
            train_sampler.set_epoch(0)


        train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4)
        test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)

        train_sampler = train_sampler
        train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=4, sampler=train_sampler)
        test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
        # classes = ('plane', 'car', 'bird', 'cat',
        #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        class_num = 10

        train_data_num = len(train_dataset)
        test_data_num = len(test_dataset)

        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()

        for client_index in range(client_number):
            train_data_local_dict[client_index] = train_dl
            test_data_local_dict[client_index] = test_dl
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            data_local_num_dict[client_index] = train_data_num // client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))
    elif args.mode == 'standalone':
        data_local_num_dict = dict()
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        for client_index in range(client_number):
            train_sampler = None
            shuffle = True
            if client_number > 1:
                train_sampler = data.distributed.DistributedSampler(
                    train_dataset, num_replicas=client_number, rank=client_index)
                train_sampler.set_epoch(0)
                shuffle = False

                # Note that test_sampler is for distributed testing to accelerate training
                test_sampler = data.distributed.DistributedSampler(
                    test_dataset, num_replicas=client_number, rank=client_index)
                train_sampler.set_epoch(0)


            train_data_global = data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=4)
            test_data_global = data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)

            train_sampler = train_sampler
            train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=shuffle, num_workers=4, sampler=train_sampler)
            test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
            # classes = ('plane', 'car', 'bird', 'cat',
            #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            class_num = 10

            train_data_num = len(train_dataset)
            test_data_num = len(test_dataset)

            train_data_local_dict[client_index] = train_dl
            test_data_local_dict[client_index] = test_dl
            # Because the train_dataset has all samples, so here we divide it to get the length of local dataset.
            data_local_num_dict[client_index] = train_data_num // client_number
            logging.info("client_index = %d, local_sample_number = %d" % (client_index, train_data_num))
    else:
        raise NotImplementedError


    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num








