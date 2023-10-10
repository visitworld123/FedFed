import logging
import random
import math
import functools
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    SVHN,
    FashionMNIST,
    MNIST,
)


from .cifar10.datasets import CIFAR10_truncated_WO_reload
from .cifar100.datasets import CIFAR100_truncated_WO_reload
from .SVHN.datasets import SVHN_truncated_WO_reload
from .FashionMNIST.datasets import FashionMNIST_truncated_WO_reload

from .cifar10.datasets import data_transforms_cifar10
from .cifar100.datasets import data_transforms_cifar100
from .SVHN.datasets import data_transforms_SVHN
from .FashionMNIST.datasets import data_transforms_fmnist


from data_preprocessing.utils.stats import record_net_data_stats



NORMAL_DATASET_LIST = ["cifar10", "cifar100", "SVHN",
                        "mnist", "fmnist", "femnist-digit", "Tiny-ImageNet-200"]



class Data_Loader(object):

    full_data_obj_dict = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "SVHN": SVHN,
        "fmnist": FashionMNIST,

    }
    sub_data_obj_dict = {
        "cifar10": CIFAR10_truncated_WO_reload,
        "cifar100": CIFAR100_truncated_WO_reload,
        "SVHN": SVHN_truncated_WO_reload,
        "fmnist": FashionMNIST_truncated_WO_reload,
    }

    transform_dict = {
        "cifar10": data_transforms_cifar10,
        "cifar100": data_transforms_cifar100,
        "SVHN": data_transforms_SVHN,
        "fmnist": data_transforms_fmnist,

    }

    num_classes_dict = {
        "cifar10": 10,
        "cifar100": 100,
        "SVHN": 10,
        "fmnist": 10,
    }


    image_resolution_dict = {
        "cifar10": 32,
        "cifar100": 32,
        "SVHN": 32,
        "fmnist": 32,
    }


    def __init__(self, args=None, process_id=0, mode="centralized", task="centralized",
                data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.5, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default", other_params={}):

        # less use this.
        self.args = args

        # For partition
        self.process_id = process_id
        self.mode = mode
        self.task = task
        self.data_efficient_load = data_efficient_load 
        self.dirichlet_balance = dirichlet_balance
        self.dirichlet_min_p = dirichlet_min_p

        self.dataset = dataset
        self.datadir = datadir
        self.partition_method = partition_method
        self.partition_alpha = partition_alpha
        self.client_number = client_number
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_sampler = data_sampler

        self.augmentation = augmentation
        self.other_params = other_params

        # For image
        self.resize = resize

        self.init_dataset_obj()




    def load_data(self):
        self.federated_standalone_split() 
        self.other_params["train_cls_local_counts_dict"] = self.train_cls_local_counts_dict
        self.other_params["client_dataidx_map"] = self.client_dataidx_map


        return self.train_data_global_num, self.test_data_global_num, self.train_data_global_dl, self.test_data_global_dl, \
               self.train_data_local_num_dict, self.test_data_local_num_dict, self.test_data_local_dl_dict, self.train_data_local_ori_dict,self.train_targets_local_ori_dict,\
               self.class_num, self.other_params


    def init_dataset_obj(self):
        self.full_data_obj = Data_Loader.full_data_obj_dict[self.dataset]
        self.sub_data_obj = Data_Loader.sub_data_obj_dict[self.dataset]
        logging.info(f"dataset augmentation: {self.augmentation}, resize: {self.resize}")
        self.transform_func = Data_Loader.transform_dict[self.dataset]  # 生成transform的function
        self.class_num = Data_Loader.num_classes_dict[self.dataset]
        self.image_resolution = Data_Loader.image_resolution_dict[self.dataset]



    def get_transform(self, resize, augmentation, dataset_type, image_resolution=32):
        MEAN, STD, train_transform, test_transform = \
            self.transform_func(
                resize=resize, augmentation=augmentation, dataset_type=dataset_type, image_resolution=image_resolution)
        # if self.args.Contrastive == "SimCLR":
        return MEAN, STD, train_transform, test_transform



    def load_full_data(self):
        # For cifar10, cifar100, SVHN, FMNIST
        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "full_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        if self.dataset == "SVHN":
            train_ds = self.full_data_obj(self.datadir,  "train", download=True, transform=train_transform, target_transform=None)
            test_ds = self.full_data_obj(self.datadir,  "test", download=True, transform=test_transform, target_transform=None)
            train_ds.data = train_ds.data.transpose((0,2,3,1))
            # test_ds.data =  test_ds.data.transpose((0,2,3,1))
            logging.info(os.getcwd())
        else:
            train_ds = self.full_data_obj(self.datadir,  train=True, download=True, transform=train_transform)
            test_ds = self.full_data_obj(self.datadir,  train=False, download=True, transform=test_transform)
            logging.info(os.getcwd())
        # X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
        # X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets

        return train_ds, test_ds  #Complete Dataset

# got it 加载不同子数据集
    def load_sub_data(self, client_index, train_ds, test_ds):

        # Maybe only ``federated`` needs this.
        dataidxs = self.client_dataidx_map[client_index]
        train_data_local_num = len(dataidxs)

        MEAN, STD, train_transform, test_transform = self.get_transform(
            self.resize, self.augmentation, "sub_dataset", self.image_resolution)

        logging.debug(f"Train_transform is {train_transform} Test_transform is {test_transform}")
        train_ds_local = self.sub_data_obj(self.datadir, dataidxs=dataidxs, train=True, transform=train_transform,
                full_dataset=train_ds)

        # get the original data without transforms, so it's in [0, 255] np array rather than Tensor
        train_ori_data = np.array(train_ds_local.data)
        train_ori_targets = np.array(train_ds_local.targets)
        test_ds_local = self.sub_data_obj(self.datadir, train=False, transform=test_transform,
                        full_dataset=test_ds)   

        test_data_local_num = len(test_ds_local)
        return train_ds_local, test_ds_local, train_ori_data, train_ori_targets, train_data_local_num, test_data_local_num

    def get_dataloader(self, train_ds, test_ds,shuffle=True, drop_last=False, train_sampler=None, num_workers=1):
        logging.info(f"shuffle: {shuffle}, drop_last:{drop_last}, train_sampler:{train_sampler} ")
        train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=shuffle,   # dl means dataloader
                                drop_last=drop_last, sampler=train_sampler, num_workers=num_workers) # sampler定义自己的sampler策略，如果指定这个参数，则shuffle必须为False
        test_dl = data.DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=True,
                                drop_last=False, num_workers=num_workers)  # drop_last为True剩余的数据不够一个batch会扔掉

        return train_dl, test_dl


    def get_y_train_np(self, train_ds):
        if self.dataset in ["fmnist"]:
            y_train = train_ds.targets.data
        elif self.dataset in ["SVHN"]:
            y_train = train_ds.labels
        else:
            y_train = train_ds.targets
        y_train_np = np.array(y_train)
        return y_train_np


    def federated_standalone_split(self):

        train_ds, test_ds = self.load_full_data()
        y_train_np = self.get_y_train_np(train_ds)  

        self.train_data_global_num = y_train_np.shape[0]
        self.test_data_global_num = len(test_ds) 

        self.client_dataidx_map, self.train_cls_local_counts_dict = self.partition_data(y_train_np, self.train_data_global_num)

        logging.info("train_cls_local_counts_dict = " + str(self.train_cls_local_counts_dict))

        self.train_data_global_dl, self.test_data_global_dl = self.get_dataloader(   # train_data_global_dataloader and test_data_global_dataloader
                train_ds, test_ds,   
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)
        logging.info("train_dl_global number = " + str(len(self.train_data_global_dl)))
        logging.info("test_dl_global number = " + str(len(self.test_data_global_dl)))



        self.train_data_local_num_dict = dict()  
        self.test_data_local_num_dict = dict()
        self.train_data_local_ori_dict = dict()
        self.train_targets_local_ori_dict = dict()
        self.test_data_local_dl_dict = dict()

        for client_index in range(self.client_number):

            train_ds_local, test_ds_local, train_ori_data, train_ori_targets, \
            train_data_local_num, test_data_local_num = self.load_sub_data(client_index, train_ds, test_ds)

            self.train_data_local_num_dict[client_index] = train_data_local_num
            self.test_data_local_num_dict[client_index] = test_data_local_num
            logging.info("client_ID = %d, local_train_sample_number = %d, local_test_sample_number = %d" % \
                         (client_index, train_data_local_num, test_data_local_num))



            train_data_local_dl, test_data_local_dl = self.get_dataloader(train_ds_local, test_ds_local,
                                                                          shuffle=True, drop_last=False, num_workers=self.num_workers)
            logging.info("client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_index, len(train_data_local_dl), len(test_data_local_dl))) # 每个local client有多少batch的数据

            self.test_data_local_dl_dict[client_index] = test_data_local_dl
            self.train_data_local_ori_dict[client_index] = train_ori_data
            self.train_targets_local_ori_dict[client_index] = train_ori_targets
            self.test_data_local_dl_dict[client_index] = test_data_local_dl




    # centralized loading
    def load_centralized_data(self):
        self.train_ds, self.test_ds = self.load_full_data()
        self.train_data_num = len(self.train_ds)
        self.test_data_num = len(self.test_ds)
        self.train_dl, self.test_dl = self.get_dataloader(
                self.train_ds, self.test_ds,
                shuffle=True, drop_last=False, train_sampler=None, num_workers=self.num_workers)






    def partition_data(self, y_train_np, train_data_num):
        logging.info("partition_method = " + (self.partition_method))
        if self.partition_method in ["homo", "iid"]:
            total_num = train_data_num
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            client_dataidx_map = {i: batch_idxs[i] for i in range(self.client_number)}

 
        elif self.partition_method == "hetero":
            min_size = 0
            K = self.class_num    
            N = y_train_np.shape[0] 
            logging.info("N = " + str(N))
            client_dataidx_map = {}  
            while min_size < self.class_num:
                idx_batch = [[] for _ in range(self.client_number)]
              
                for k in range(K): 
                    idx_k = np.where(y_train_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.client_number))
                    if self.dirichlet_balance:
                        argsort_proportions = np.argsort(proportions, axis=0)
                        if k != 0:
                            used_p = np.array([len(idx_j) for idx_j in idx_batch])
                            argsort_used_p = np.argsort(used_p, axis=0)
                            inv_argsort_proportions = argsort_proportions[::-1]
                            proportions[argsort_used_p] = proportions[inv_argsort_proportions]
                    else:
                        proportions = np.array([p * (len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])

                    ## set a min value to smooth, avoid too much zero samples of some classes.
                    if self.dirichlet_min_p is not None:
                        proportions += float(self.dirichlet_min_p)
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]

        elif self.partition_method > "noniid-#label0" and self.partition_method <= "noniid-#label9":
            num = eval(self.partition_method[13:])
            if self.dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
                num = 1
                K = 2
            else:
                K = self.class_num
            if num == 10:
                client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(10):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.client_number)
                    for j in range(self.client_number):
                        client_dataidx_map[j]=np.append(client_dataidx_map[j],split[j])
            else:
                times=[0 for i in range(10)]
                contain=[]
                for i in range(self.client_number):
                    current=[i%K]
                    times[i%K]+=1
                    j=1
                    while (j<num):
                        ind=random.randint(0,K-1)
                        if (ind not in current):
                            j=j+1
                            current.append(ind)
                            times[ind]+=1
                    contain.append(current)
                client_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(self.client_number)}
                for i in range(K):
                    idx_k = np.where(y_train_np==i)[0]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k,times[i])
                    ids=0
                    for j in range(self.client_number):
                        if i in contain[j]:
                            client_dataidx_map[j]=np.append(client_dataidx_map[j],split[ids])
                            ids+=1

        elif self.partition_method == "long-tail":
            if self.client_number == 10 or self.client_number == 100:
                pass
            else:
                raise NotImplementedError

            main_prop = self.partition_alpha / (self.client_number // self.class_num)


            tail_prop = (1 - main_prop) / (self.client_number - self.client_number // self.class_num)

            client_dataidx_map = {}
    
            K = self.class_num
            idx_batch = [[] for _ in range(self.client_number)]
            for k in range(K):
                idx_k = np.where(y_train_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.array([tail_prop for _ in range(self.client_number)])
                main_clients = np.array([k + i * K for i in range(self.client_number // K)])
                proportions[main_clients] = main_prop
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                client_dataidx_map[j] = idx_batch[j]
        if self.partition_method == "hetero-fix":
            pass

        else:
            train_cls_local_counts_dict = record_net_data_stats(y_train_np, client_dataidx_map)

        return client_dataidx_map, train_cls_local_counts_dict













