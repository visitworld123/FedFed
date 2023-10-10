import logging

import numpy as np

# from data_preprocessing.generative_loader import Generative_Data_Loader
generative_image_resolution_dict = {
    "style_GAN_init": 32,
    "style_GAN_init_32_c100": 32,
    "style_GAN_init_32_c62": 32,
    "style_GAN_init_64_c200": 64,
    "Gaussian_Noise": 32,
    "cifar_conv_decoder": 32,
}

def record_batch_data_stats(y_train, bs=None, num_classes=10):
    if bs is not None:
        bs = y_train.shape[0]

    batch_cls_counts = {}
    for i in range(num_classes):
        num_label = (y_train == i).sum().item()
        batch_cls_counts[i] = num_label
    # logging.debug('Batch Data statistics: %s' % str(batch_cls_counts))
    return batch_cls_counts

# got it 数据的label和partition不同client的dataidx_map
# 返回net_cls_counts不同client有哪几个类，这几个类分别有，net_cls_counts中每一个key代表一个client，value还是一个dict，key是该client包含的类别，value是该类别个数
def record_net_data_stats(y_train, net_dataidx_map):
    client_train_cls_counts_dict = {}

    for client_idx, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)   # 不同client有几个类，该类分别出现了几次
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_train_cls_counts_dict[client_idx] = tmp
    logging.debug('Data statistics: %s' % str(client_train_cls_counts_dict))
    return client_train_cls_counts_dict



def get_dataset_image_size(dataset):
    if dataset in ["cifar10", "cifar100", "SVHN"]:
        image_size = 32
    elif dataset in ["mnist", "fmnist", "femnist", "femnist-digit"]:
        image_size = 28
    elif dataset in ["Tiny-ImageNet-200"]:
        image_size = 64
    elif dataset in generative_image_resolution_dict:
        image_size = generative_image_resolution_dict[dataset]
    else:
        logging.info(f"Input dataset: {dataset}, not found")
        raise NotImplementedError
    return image_size








