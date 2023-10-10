import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST
import torch
import torchvision.transforms as transforms

from data_preprocessing.utils.utils import Cutout

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def data_transforms_fmnist(resize=28, augmentation="default", dataset_type="full_dataset",
                            image_resolution=32):
    train_transform = transforms.Compose([])
    test_transform = transforms.Compose([])

    if dataset_type == "full_dataset":
        pass
    elif dataset_type == "sub_dataset":
        pass
    else:
        raise NotImplementedError

    if resize == 28:
        pass
    else:
        image_size = resize
        train_transform.transforms.append(transforms.Resize(resize))
        test_transform.transforms.append(transforms.Resize(resize))

    if augmentation == "default":
        train_transform.transforms.append(transforms.RandomCrop(image_size, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        #train_transform.transforms.append(RandAugmentMC(n=2, m=10))
    else:
        raise NotImplementedError

    train_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.ToTensor())

    return None, None, train_transform, test_transform





class FashionMNIST_truncated(data.Dataset):

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
        mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        # print("train member of the class: {}".format(self.train))
        # data = cifar_dataobj.train_data
        data = mnist_dataobj.data
        targets = mnist_dataobj.targets
        # targets = np.array(mnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    # def truncate_channel(self, index):
    #     for i in range(index.shape[0]):
    #         gs_index = index[i]
    #         self.data[gs_index, :, :, 1] = 0.0
    #         self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)



class FashionMNIST_truncated_WO_reload(data.Dataset):

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
        # print("download = " + str(self.download))
        # mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        if self.train:
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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)



