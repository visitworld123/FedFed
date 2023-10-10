import logging

import torch

import numpy as np


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, args, dataset, indices=None, num_samples=None, class_num=10, **kwargs):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        logging.info("self.indices: {}".format(self.indices))
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        self.args = args
        self.dataset = dataset
        # distribution of classes in the dataset 
        # label_to_count = [0] * len(np.unique(dataset.target))
        label_to_count = [0] * class_num
        logging.info("label_to_count: {}".format(label_to_count))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        for i in range(len(label_to_count)):
            if label_to_count[i] == 0:
                label_to_count[i] = 1

        self.label_to_count = label_to_count

        effective_num = 1.0 - np.power(self.beta, label_to_count)
        per_cls_weights = (1.0 - self.beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.target[idx]

    def update(self, **kwargs):
        pass


    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples