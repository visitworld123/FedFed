import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import random
import argparse

from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from collections import OrderedDict

import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

# from data_preprocessing.pascal_voc_augmented.data_loader import partition_data as partition_pascal
# from data_preprocessing.cifar100.data_loader import partition_data as partition_cifar100
# from data_preprocessing.cifar10.data_loader import partition_data as partition_cifar10
# from data_preprocessing.FashionMNIST.data_loader import partition_data as partition_fmnist
# from data_preprocessing.MNIST.data_loader import partition_data as partition_mnist
# from data_preprocessing.TinyImageNet.data_loader import partition_data as partition_Tiny_ImageNet_200

from data_preprocessing.build import load_data


from data_preprocessing.loader import Data_Loader
from data_preprocessing.loader_shakespeare import Shakespeare_Data_Loader
from data_preprocessing.generative_loader import Generative_Data_Loader

from data_preprocessing.loader import NORMAL_DATASET_LIST
from data_preprocessing.loader_shakespeare import SHAKESPEARE_DATASET_LIST
from data_preprocessing.generative_loader import GENERATIVE_DATASET_LIST
from data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist


def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition_method', type=str, default="hetero", help='Partition Alpha')
    parser.add_argument('--data_efficient_load', type=bool, default=True, help='')
    parser.add_argument('--dirichlet_min_p', type=str, default=None, help='Partition Alpha')
    parser.add_argument('--TwoCropTransform', type=str, default=False)

    parser.add_argument('--dirichlet_balance', type=bool, default=False, help='Partition Alpha')

    parser.add_argument('--seed', type=int, default=0, help="random seed")

    parser.add_argument('--alpha', type=float, default=0.5, help='Partition Alpha')
    parser.add_argument('--data_dir', type=str, default='~/datasets/cifar100', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='pascal_voc', help="Name of dataset")
    parser.add_argument('--plus_common_data', type=bool, default=False, help="Name of dataset")
    parser.add_argument('--client_num_in_total', type=int, default=100,
                        help='Number of total clients')

    args = parser.parse_args()

    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True

    partition_method = args.partition_method
    alpha = args.alpha
    client_num = args.client_num_in_total
    # classes = list(range(1, 20 + 1, 1))

    data_dir = args.data_dir
    if args.dataset == 'pascal_voc':
        net_data_idx_map, train_data_cls_counts = partition_pascal(data_dir, partition_method, client_num, alpha, 513)
    # elif args.dataset == 'cifar100':
    #     classes = list(range(100))
    #     _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar100_train_ds, cifar100_test_ds = partition_cifar100(
    #         'cifar100', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    # elif args.dataset == 'cifar10':
    #     classes = list(range(10))
    #     _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar10_train_ds, cifar10_test_ds = partition_cifar10(
    #         'cifar10', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
        # print("!!!!!!", train_data_cls_counts)
    elif args.dataset == 'gld23k':
        classes = list(range(203))
        client_num = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')
        net_data_idx_map, train_data_cls_counts =partition_gld(
            args.dataset, args.data_dir, fed_train_map_file, fed_test_map_file)
    # elif args.dataset == 'fmnist':
    #     classes = list(range(10))
    #     _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar10_train_ds, cifar10_test_ds = partition_fmnist(
    #         'fmnist', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    #     # print("!!!!!!", train_data_cls_counts)
    # elif args.dataset == 'mnist':
    #     _, _, _, _, net_data_idx_map, train_data_cls_counts= partition_mnist(
    #         'mnist', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    elif args.dataset == 'Tiny-ImageNet-200':
        classes = list(range(200))
        _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar100_train_ds, cifar100_test_ds= partition_Tiny_ImageNet_200(
            'Tiny-ImageNet-200', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    elif args.dataset in ['Tiny-ImageNet-200', 'cifar10', 'cifar100', 'mnist', 'fmnist']:
        data_loader = Data_Loader(args, process_id=0, mode="standalone", task="federated", data_efficient_load=True, dirichlet_balance=False, dirichlet_min_p=None,
            dataset=args.dataset, datadir=args.data_dir, partition_method=args.partition_method, partition_alpha=args.alpha,
            client_number=args.client_num_in_total, batch_size=64, num_workers=1,
            data_sampler='Random',
            other_params={})
        train_ds, test_ds = data_loader.load_full_data()
        # y_train = train_ds.targets
        # y_train_np = np.array(y_train)
        y_train_np = data_loader.get_y_train_np(train_ds)
        # class_num = len(np.unique(y_train))
        train_data_num = y_train_np.shape[0]
        classes = list(range(data_loader.class_num))
        net_data_idx_map, train_data_cls_counts = data_loader.partition_data(y_train_np, train_data_num)
    else:
        raise NotImplementedError



    print(train_data_cls_counts, classes)
    print("============================================")
    all_data = sum([len(value) for value in net_data_idx_map.values()])
    print([f" {key}: {len(value)}" for key, value in net_data_idx_map.items()])
    print(f"all data num: {all_data}")
    print("============================================")

    # Adding missing classes to list
    for key in train_data_cls_counts:
        if len(classes) != len(train_data_cls_counts[key]):
            # print(len(classes))
            # print(len(train_data_cls_counts[key]))
            add_classes = set(classes) - set(train_data_cls_counts[key])
            # print(add_classes)
            for e in add_classes:
                train_data_cls_counts[key][e] = 0

    clients = list(range(client_num))

    # Sort the class key values to easily convert to array while preserving order
    samples = []
    for key in train_data_cls_counts:
        od = OrderedDict(sorted(train_data_cls_counts[key].items()))
        if args.plus_common_data:
            samples.append(list(od.values())+[int(50000 / len(classes))  for _ in classes])
        else:
            samples.append(list(od.values()))

    # if args.plus_common_data:
    #     new_train_data_cls_counts = {}
    #     for client in train_data_cls_counts:
    #         new_train_data_cls_counts[client] = {}
    #         for class_i in classes:
    #             new_train_data_cls_counts[client][class_i+len(classes)] = 5000
    #         od = OrderedDict(sorted(train_data_cls_counts[key].items()))
    #         samples.append(list(od.values()))
    #     num_classes = len(classes)
    #     for i in range(num_classes):
    #         classes.append(i+num_classes)

    data = np.array(samples)
    transpose_data = data.T

    fig, ax = plt.subplots()

    print(transpose_data, classes, clients)
    # if args.dataset == 'cifar100':
    #     classes = list(range(0, 100, 20))
    #     clients = list(range(0, 100, 20))
    #     classes.append(99)
    #     clients.append(99)
    # elif args.dataset == 'gld23k':
    #     classes = list(range(0, 203, 40))
    #     clients = list(range(0, 233, 40))
    #     clients.append(232)
    # elif args.dataset == 'cifar10':
    #     classes = list(range(0, 10))
    #     clients = list(range(0, 10))
    # elif args.dataset == 'fmnist':
    #     classes = list(range(0, 10))
    #     clients = list(range(0, 10))
    # elif args.dataset == 'mnist':
    #     classes = list(range(0, 10))
    #     clients = list(range(0, 10))

    if client_num == 10:
        clients = list(range(0, 10))
    elif client_num == 20:
        clients = list(range(0, 20))
    elif client_num == 100:
        clients = list(range(0, 100, 20))
    else:   
        raise NotImplementedError

    if len(classes) == 10:
        classes = list(range(0, 10))
    elif len(classes) == 20:
        classes = list(range(0, 20, 2))
    elif len(classes) == 100:
        classes = list(range(0, 100, 20))
    elif len(classes) == 200:
        classes = list(range(0, 200, 40))
    else:   
        raise NotImplementedError


    fontsize = 18
    # im, cbar = heatmap(transpose_data, classes, clients, classes, clients, ax=ax,
    #                    fontsize=fontsize, cmap="YlGn", cbarlabel="samples")
    # annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=5000,
    #                 textcolors=("red", "white"), dataset=args.dataset, fontsize=int(fontsize*0.6))

    sns.heatmap(transpose_data, cmap='viridis')

    ax = plt.gca()
    # ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_ylabel('Class ID')
    ax.set_xlabel('Party ID')


    # if client_num == len(classes):
    fig.set_figheight(5)
    fig.set_figwidth(7)

    fig.tight_layout()

    plt.show()


    if args.plus_common_data:
        file_name = "WithCommon_" + args.dataset + '_partition' + str(partition_method) + \
            '_alpha' + str(alpha) + '_DirMin' + str(args.dirichlet_min_p) + \
            '_clients' + str(client_num)

    else:
        file_name = args.dataset + '_partition' + str(partition_method) + \
            '_alpha' + str(alpha) + '_DirMin' + str(args.dirichlet_min_p) + \
            '_clients' + str(client_num)

    plt.savefig(file_name + '.pdf')

