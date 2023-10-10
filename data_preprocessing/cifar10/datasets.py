import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10
import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout
from utils.randaugment4fixmatch import RandAugmentMC

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


# got it 创建data_transform,resize需要变到的大小和是否需要augmentation
# 返回Normalize的mean和std还有对train的transform和test的transform
def data_transforms_cifar10(resize=32, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    image_size = 32

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        train_transform.transforms.append(transforms.ToPILImage())
    else:
        raise NotImplementedError

    if resize is 32:
        pass
    else:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        train_transform.transforms.append(RandAugmentMC(n=2, m=10))
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    #train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    if augmentation == "default":
        pass
        # train_transform.transforms.append(Cutout(16))
    elif augmentation == "no":
        pass
    else:
        raise NotImplementedError

    test_transform.transforms.append(transforms.ToTensor())
    #test_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    return CIFAR_MEAN, CIFAR_STD, train_transform, test_transform






class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data[self.dataidxs]
            targets = np.array(cifar_dataobj.targets)[self.dataidxs]
        else:
            data = cifar_dataobj.data
            targets = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets


    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)




class CIFAR10_truncated_WO_reload(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
       
        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = self.full_dataset.data[self.dataidxs]
            targets = np.array(self.full_dataset.targets)[self.dataidxs]
        else:
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.targets)



        return data, targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


# part of CIFAR10 data from a sample algorithm
class Dataset_Personalize(data.Dataset):

    def __init__(self, data, targets, transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class Dataset_Personalize_4Tensor(data.Dataset):

    def __init__(self, data, targets, transform=None, target_transform=None):

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class Dataset_3Types_ImageData(data.Dataset):

    def __init__(self, ori_data, share_data1,share_data2, ori_targets, share_targets1,
                 share_targets2, transform=None, share_transform=None, target_transform=None):

        # ori_data is numpy from original data without transform and is not a tensor
        # share_data is shared by all clients, so its tensor
        self.ori_data = ori_data
        self.share_data1 = share_data1
        self.ori_targets = ori_targets
        self.share_targets1 = share_targets1
        self.share_data2 = share_data2
        self.share_targets2 = share_targets2
        self.transform = transform
        self.share_transform = share_transform
        self.target_transform = target_transform



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img1, img2, img3, targets1, targets2, targets3 = \
        self.ori_data[index], self.share_data1[index], self.share_data2[index],\
        self.ori_targets[index], self.share_targets1[index], self.share_targets2[index]

        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img1 = self.transform(img1)

        if self.share_transform is not None:
            img2 = self.share_transform(img2)
            img3 = self.share_transform(img3)

        if self.target_transform is not None:
            targets1 = self.target_transform(targets1)
            targets2 = self.target_transform(targets2)
            targets3 = self.target_transform(targets3)

        return img1, img2, img3, targets1, targets2, targets3

    def __len__(self):
        if len(self.ori_data) == len(self.share_data1):
            return len(self.ori_data)
        else:
            raise RuntimeError("shared data {} is not equal to ori_data {}".format(len(self.share_data1), len(self.ori_data)))



