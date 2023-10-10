import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import SVHN
import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')



def data_transforms_SVHN(resize=32, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    SVHN_MEAN = [0.5, 0.5, 0.5]
    SVHN_STD = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        pass
    else:
        raise NotImplementedError

    if resize == 32:
        pass
    else:
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        pass
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.ToTensor())

    return SVHN_MEAN, SVHN_STD, train_transform, train_transform


class SVHN_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        # def target_transform(target):
        #     return int(target) - 1

        self.target_transform = target_transform

        self.download = download

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if self.train:
            SVHN_dataobj = SVHN(self.root, "train", self.transform, self.target_transform, self.download)
        else:
            SVHN_dataobj = SVHN(self.root, "test", self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = SVHN_dataobj.train_data
            data = SVHN_dataobj.data
            targets = np.array(SVHN_dataobj.labels)
        else:
            data = SVHN_dataobj.data
            targets = np.array(SVHN_dataobj.labels)

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




class SVHN_truncated_WO_reload(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None,
                full_dataset=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        # def target_transform(target):
        #     return int(target) - 1
        self.target_transform = target_transform
        self.full_dataset = full_dataset

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        # print("download = " + str(self.download))
        # SVHN_dataobj = SVHN(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = self.full_dataset.data[self.dataidxs]
            targets = np.array(self.full_dataset.labels)[self.dataidxs]
        else:
            data = self.full_dataset.data
            targets = np.array(self.full_dataset.labels)

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
        # img, target = self.data[index], self.target[index]
        # print("svhn img:", img)
        # print("svhn target:", target)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class SVHN_Personalize_4Tensor(data.Dataset):

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





