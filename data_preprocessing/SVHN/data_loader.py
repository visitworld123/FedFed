import logging
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

from .datasets import SVHN_truncated, SVHN_truncated_WO_reload

from data_preprocessing.utils.imbalance_data import ImbalancedDatasetSampler


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/SVHN/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/SVHN/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_SVHN():
    SVHN_MEAN = [0.5, 0.5, 0.5]
    SVHN_STD = [0.5, 0.5, 0.5]

    # train_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(SVHN_MEAN, SVHN_STD),
    # ])
    # train_transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    train_transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    def target_transform(target):
        return int(target) - 1

    return train_transform, train_transform


def load_SVHN_data(datadir, args=None):
    train_transform, test_transform = _data_transforms_SVHN()

    # def target_transform(target):
    #     return int(target) - 1
    target_transform = None

    if args.data_efficient_load:
        SVHN_train_ds = SVHN(datadir, "train", transform=train_transform, target_transform=target_transform, download=True)
        SVHN_test_ds = SVHN(datadir, "test", transform=test_transform, target_transform=target_transform, download=True)
    else:
        SVHN_train_ds = SVHN_truncated(datadir, train=True, download=True, transform=train_transform, target_transform=target_transform)
        SVHN_test_ds = SVHN_truncated(datadir, train=False, download=True, transform=test_transform, target_transform=target_transform)

    X_train, y_train = SVHN_train_ds.data, SVHN_train_ds.labels
    X_test, y_test = SVHN_test_ds.data, SVHN_test_ds.labels

    return (X_train, y_train, X_test, y_test, SVHN_train_ds, SVHN_test_ds)


def partition_data(dataset, datadir, partition, n_nets, alpha, args=None):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test, SVHN_train_ds, SVHN_test_ds = load_SVHN_data(
        datadir, args)

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    n_train = X_train_np.shape[0]
    # n_test = X_test.shape[0]

    if partition in ["homo", "iid"]:
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 10
        N = y_train_np.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train_np == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                # used_p = np.array([len(idx_j) for idx_j in idx_batch])
                # if used_p.sum() > 0:
                #     used_p = used_p / used_p.sum()
                # else:
                #     used_p = np.array([1 for i in range(n_nets)])
                #     used_p = used_p / used_p.sum()
                # used_p = 1 - used_p
                # used_p = used_p / used_p.sum()
                # # proportions = proportions + 0.5*used_p
                # proportions = proportions + 5*used_p
                # proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                # proportions = np.array([p * ( p*5000 + len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                if args.dirichlet_balance:
                    argsort_proportions = np.argsort(proportions, axis=0)
                    if k != 0:
                        used_p = np.array([len(idx_j) for idx_j in idx_batch])
                        argsort_used_p = np.argsort(used_p, axis=0)
                        inv_argsort_proportions = argsort_proportions[::-1]
                        # print(used_p)
                        # print(argsort_used_p)
                        # proportions = np.random.random(n_nets)
                        proportions[argsort_used_p] = proportions[inv_argsort_proportions]
                        # print(np.argsort(proportions, axis=0))
                else:
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])

                ## set a min value to smooth, avoid too much zero samples of some classes.
                if args.dirichlet_min_p is not None:
                    proportions += float(args.dirichlet_min_p)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # refer to https://github.com/Xtra-Computing/NIID-Bench/blob/main/utils.py
    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
            num = 1
            K = 2
        else:
            K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(10):
                idx_k = np.where(y_train_np==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, n_nets)
                for j in range(n_nets):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(n_nets):
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
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_nets)}
            for i in range(K):
                idx_k = np.where(y_train_np==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_nets):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1
    elif partition == "long-tail":
        if n_nets == 10 or n_nets == 100:
            pass
        else:
            raise NotImplementedError
        
        # There are  n_nets // 10 clients share the \alpha proportion of data of one class
        main_prop = alpha / (n_nets // 10)

        # There are (n_nets - n_nets // 10) clients share the tail of one class
        tail_prop = (1 - main_prop) / (n_nets - n_nets // 10)

        net_dataidx_map = {}
        # for each class in the dataset
        K = 10
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train_np == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.array([ tail_prop for _ in range(n_nets)])
            main_clients = np.array([ k + i*K for i in range(n_nets // K)])
            proportions[main_clients] = main_prop
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]


    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/SVHN/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/SVHN/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train_np, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
                SVHN_train_ds, SVHN_test_ds


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, args=None,
                        full_train_dataset=None, full_test_dataset=None):
    return get_dataloader_SVHN(datadir, train_bs, test_bs, dataidxs, args=args,
                        full_train_dataset=full_train_dataset, full_test_dataset=full_test_dataset)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_SVHN(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_SVHN(datadir, train_bs, test_bs, dataidxs=None, args=None,
                        full_train_dataset=None, full_test_dataset=None):

    transform_train, transform_test = _data_transforms_SVHN()
    # def target_transform(target):
    #     return int(target) - 1
    target_transform = None

    if args.data_efficient_load:
        dl_obj = SVHN_truncated_WO_reload
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                        target_transform=target_transform,
                        full_dataset=full_train_dataset)
        test_ds = dl_obj(datadir, train=False, transform=transform_test,
                        target_transform=target_transform,
                        full_dataset=full_test_dataset)
    else:
        dl_obj = SVHN_truncated

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train,
                        target_transform=target_transform,
                        download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test,
                        target_transform=target_transform,
                        download=True)

    drop_last = True
    if args.batch_size > len(train_ds):
        drop_last = False

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=drop_last)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=drop_last)

    return train_dl, test_dl


def get_dataloader_test_SVHN(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = SVHN_truncated

    transform_train, transform_test = _data_transforms_SVHN()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_SVHN(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size, args=None):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
        SVHN_train_ds, SVHN_test_ds = partition_data(dataset,
                                            data_dir,
                                            partition_method,
                                            client_number,
                                            partition_alpha,
                                            args)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_SVHN(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,
                                args=None):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts, \
        SVHN_train_ds, SVHN_test_ds = partition_data(dataset,
                                            data_dir,
                                            partition_method,
                                            client_number,
                                            partition_alpha,
                                            args)

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        dataset, data_dir, batch_size, batch_size, args=args,
        full_train_dataset=SVHN_train_ds,
        full_test_dataset=SVHN_test_ds
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_index in range(client_number):
        dataidxs = net_dataidx_map[client_index]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_index] = local_data_num
        logging.info("client_index = %d, local_sample_number = %d" % (client_index, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                dataidxs, args=args,
                                                full_train_dataset=SVHN_train_ds,
                                                full_test_dataset=SVHN_test_ds
                                            )
        logging.info("client_index = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_index, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_index] = train_data_local
        test_data_local_dict[client_index] = test_data_local
    # return train_data_num, test_data_num, train_data_global, test_data_global, \
    #        data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts
    return train_data_num, test_data_num, train_data_global, test_data_global, \
        data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num




