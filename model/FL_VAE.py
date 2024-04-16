from __future__ import print_function
import abc
import os
import math

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import sys
sys.path.append('.')
sys.path.append('..')
from utils.normalize import *
from model.cv.others import (ModerateCNNMNIST, ModerateCNN)
from model.cv.resnet_v2 import ResNet18
from utils.log_info import *
import torchvision.transforms as transforms


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, norm=False):  # usually Wide_ResNet(28,10,0.3,10)
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.normalize = CIFARNORMALIZE(32)
        self.norm = norm

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.norm:
            x = self.normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class FL_CVAE_cifar(AbstractAutoEncoder):
    def __init__(self, args, d, z, device, with_classifier=True, **kwargs):
        super(FL_CVAE_cifar, self).__init__()

        # if args.dataset == 'fmnist':
        #     self.init_bn = nn.BatchNorm2d(1)
        # else:
        #     self.init_bn = nn.BatchNorm2d(3)
        self.noise_mean = args.VAE_mean
        self.noise_std1 = args.VAE_std1
        self.noise_std2 = args.VAE_std2
        self.device = device
        self.noise_type = args.noise_type
        self.encoder_former = nn.Conv2d(1, d // 2, kernel_size=4, stride=2, padding=1, bias=False) if args.dataset == 'fmnist' else \
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.decoder_last = nn.ConvTranspose2d(d // 2, 1, kernel_size=4, stride=2, padding=1, bias=False) if args.dataset == 'fmnist' else \
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False)
        if args.dataset == 'fmnist':
            self.xi_bn = nn.BatchNorm2d(1)
        else:
            self.xi_bn = nn.BatchNorm2d(3)

        self.sigmoid = nn.Sigmoid()

        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f ** 2, self.z) # 2048------>2048
        self.fc12 = nn.Linear(d * self.f ** 2, self.z) # 2048------>2048
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)  # 2048------>2048
        # constrain rx
        self.relu = nn.ReLU()

        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = ResNet18(args=args, num_classes=args.num_classes, image_size=32,model_input_channels=args.model_input_channels)

    def _add_noise(self, data, size, mean, std): #
        if self.noise_type == 'Gaussian':
            rand = torch.normal(mean=mean, std=std, size=size).to(self.device)
        if self.noise_type == 'Laplace':
            rand = torch.Tensor(np.random.laplace(loc=mean, scale=std, size=size)).to(self.device)
        data += rand
        return data

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
             return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        x_no_normalize = x
        bn_x = x
        x = self.encoder_former(bn_x)
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar) #+ noise* torch.randn(mu.size()).cuda()
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.decoder_last(xi)
        xi = self.xi_bn(xi)
        xi = self.sigmoid(xi)

        if self.with_classifier:
            size = xi[0].shape
            rx = x_no_normalize - xi
            rx_noise1 = self._add_noise(torch.clone(rx),size, self.noise_mean, self.noise_std1)
            rx_noise2 = self._add_noise(torch.clone(rx), size, self.noise_mean, self.noise_std2)
            data = torch.cat((rx_noise1, rx_noise2, bn_x), dim = 0)
            out = self.classifier(data)
            return out, hi, xi, mu, logvar, rx, rx_noise1, rx_noise2
        else:
            return xi

    def classifier_test(self, data):
        if self.with_classifier:
            out = self.classifier(data)
            return out
        else:
            raise RuntimeError('There is no Classifier')

    def get_classifier(self):
        return self.classifier
