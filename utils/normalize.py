import torch
from torch import nn

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_cifar_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        # mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i], device='cuda'))
        # std_list.append(torch.full((resol, resol), CIFAR_STD[i], device='cuda'))
        mean_list.append(torch.full((resol, resol), CIFAR_MEAN[i]))
        std_list.append(torch.full((resol, resol), CIFAR_STD[i]))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

def get_imagenet_params(resol):
    mean_list = []
    std_list = []
    for i in range(3):
        mean_list.append(torch.full((resol, resol), IMAGENET_MEAN[i], device='cuda'))
        std_list.append(torch.full((resol, resol), IMAGENET_STD[i], device='cuda'))
    return torch.unsqueeze(torch.stack(mean_list), 0), torch.unsqueeze(torch.stack(std_list), 0)

class CIFARNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class CIFARINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_cifar_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x

class IMAGENETNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x

class IMAGENETINNORMALIZE(nn.Module):
    def __init__(self, resol):
        super().__init__()
        self.mean, self.std = get_imagenet_params(resol)

    def forward(self, x):
        '''
        Parameters:
            x: input image with pixels normalized to ([0, 1] - IMAGENET_MEAN) / IMAGENET_STD
        '''
        x = x.mul(self.std)
        x = x.add(*self.mean)
        return x
