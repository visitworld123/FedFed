import matplotlib
import numpy as np
import matplotlib.pyplot as plt
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
from data_preprocessing.cifar100.data_loader import partition_data as partition_cifar100
from data_preprocessing.cifar10.data_loader import partition_data as partition_cifar10
from data_preprocessing.FashionMNIST.data_loader import partition_data as partition_fmnist
from data_preprocessing.MNIST.data_loader import partition_data as partition_mnist



def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)


# Reference: https://matplotlib.org/stable/gallery/statistics/hist.html
def heatmap(data, row_ticks_to_show, col_ticks_to_show, 
            row_labels, col_labels, ax=None, fontsize=15,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    update_fontsize(cbar.ax, fontsize)
    # cbar.ax.set_fontsize(fontsize)

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    # ax.set_xticks(col_ticks_to_show)
    # ax.set_yticks(row_ticks_to_show)
    # ... and label them with the respective list entries.
    # ax.set_xticklabels(col_labels)
    # ax.set_yticklabels(row_labels)

    #  Normalize for drawing a square
    D = max(data.shape[0], data.shape[1]) 
    ax.set_xticks(np.arange(D+1)-.5, minor=True)
    ax.set_yticks(np.arange(D+1)-.5, minor=True)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)

    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.grid(which="minor", color="r", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel('Class ID')
    ax.set_xlabel('Party ID')
    update_fontsize(ax, fontsize=fontsize)

    return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, dataset='cifar10', **textkw):
    """
    A function to annotate a heatmap.
    We're not using this for our graphs
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            if dataset == 'cifar10':
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            elif dataset == 'pascal_voc':
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            else:
                text = None
            texts.append(text)

    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition_method', type=str, default="hetero", help='Partition Alpha')
    parser.add_argument('--data_efficient_load', type=bool, default=True, help='')
    parser.add_argument('--dirichlet_min_p', type=str, default=None, help='Partition Alpha')

    parser.add_argument('--dirichlet_balance', type=bool, default=False, help='Partition Alpha')

    parser.add_argument('--alpha', type=float, default=0.5, help='Partition Alpha')
    parser.add_argument('--data_dir', type=str, default='~/datasets/cifar100', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='pascal_voc', help="Name of dataset")
    parser.add_argument('--client_num_in_total', type=int, default=100,
                        help='Number of total clients')

    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic =True

    args = parser.parse_args()

    partition_method = args.partition_method
    alpha = args.alpha
    client_num = args.client_num_in_total
    # classes = list(range(1, 20 + 1, 1))

    data_dir = args.data_dir
    if args.dataset == 'pascal_voc':
        net_data_idx_map, train_data_cls_counts = partition_pascal(data_dir, partition_method, client_num, alpha, 513)
    elif args.dataset == 'cifar100':
        classes = list(range(100))
        _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar100_train_ds, cifar100_test_ds = partition_cifar100(
            'cifar100', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    elif args.dataset == 'cifar10':
        classes = list(range(10))
        _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar10_train_ds, cifar10_test_ds = partition_cifar10(
            'cifar10', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
        # print("!!!!!!", train_data_cls_counts)
    elif args.dataset == 'gld23k':
        classes = list(range(203))
        client_num = 233
        fed_train_map_file = os.path.join(args.data_dir, 'mini_gld_train_split.csv')
        fed_test_map_file = os.path.join(args.data_dir, 'mini_gld_test.csv')
        net_data_idx_map, train_data_cls_counts =partition_gld(
            args.dataset, args.data_dir, fed_train_map_file, fed_test_map_file)
    elif args.dataset == 'fmnist':
        classes = list(range(10))
        _, _, _, _, net_data_idx_map, train_data_cls_counts, cifar10_train_ds, cifar10_test_ds = partition_fmnist(
            'fmnist', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
        # print("!!!!!!", train_data_cls_counts)
    elif args.dataset == 'mnist':
        _, _, _, _, net_data_idx_map, train_data_cls_counts= partition_mnist(
            'mnist', args.data_dir, partition_method, args.client_num_in_total, args.alpha, args)
    else:
        raise NotImplementedError


    print(train_data_cls_counts, classes)
    # Adding missing classes to list
    for key in train_data_cls_counts:
        if len(classes) != len(train_data_cls_counts[key]):
            # print(len(classes))
            # print(len(train_data_cls_counts[key]))
            add_classes = set(classes) - set(train_data_cls_counts[key])
            # print(add_classes)
            for e in add_classes:
                train_data_cls_counts[key][e] = 0

    classes = train_data_cls_counts[0].keys()
    clients = list(range(client_num))

    # Sort the class key values to easily convert to array while preserving order
    samples = []
    for key in train_data_cls_counts:
        od = OrderedDict(sorted(train_data_cls_counts[key].items()))
        samples.append(list(od.values()))
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
    elif client_num == 100:
        clients = list(range(0, 100, 20))
    else:   
        raise NotImplementedError

    if len(classes) == 10:
        classes = list(range(0, 10))
    elif len(classes) == 100:
        classes = list(range(0, 100, 20))
    else:   
        raise NotImplementedError


    fontsize = 18
    im, cbar = heatmap(transpose_data, classes, clients, classes, clients, ax=ax,
                       fontsize=fontsize, cmap="YlGn", cbarlabel="samples")
    annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=5000,
                    textcolors=("red", "white"), dataset=args.dataset, fontsize=int(fontsize*0.6))


    # if client_num == len(classes):
    fig.set_figheight(5)
    fig.set_figwidth(7)

    fig.tight_layout()

    plt.show()
    plt.savefig(args.dataset + '_partition' + str(partition_method) + \
        '_alpha' + str(alpha) + '_DirMin' + str(args.dirichlet_min_p) + \
        '_clients' + str(client_num) + '.pdf')

