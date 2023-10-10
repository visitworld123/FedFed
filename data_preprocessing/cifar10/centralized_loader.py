import os
import argparse
import time
import math
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler

from .transform import data_transforms_cifar10



def load_centralized_cifar10(dataset, data_dir, batch_size, 
                max_train_len=None, max_test_len=None,
                resize=32, augmentation=True,
                args=None):

    train_transform, test_transform = data_transforms_cifar10(resize=resize, augmentation=augmentation)

    train_dataset = CIFAR10(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR10(root=data_dir, train=False,
                            transform=test_transform, download=False)

    if max_train_len is not None:
        train_dataset.data = train_dataset.data[0: max_train_len]
        train_dataset.target = np.array(train_dataset.targets)[0: max_train_len]
    shuffle = True

    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class_num = 10

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    return train_dl, test_dl, train_data_num, test_data_num, class_num








