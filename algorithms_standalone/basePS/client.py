import logging
import math
import os
import sys
import random
from abc import  abstractmethod
from copy import deepcopy

import numpy as np
import torch

from algorithms.basePS.ps_client_trainer import PSTrainer
from utils.data_utils import optimizer_to
from model.FL_VAE import *
from optim.AdamW import AdamW
from utils.tool import *
from utils.set import *
from data_preprocessing.cifar10.datasets import Dataset_Personalize, Dataset_3Types_ImageData
import torchvision.transforms as transforms
from utils.log_info import log_info
from utils.randaugment4fixmatch import RandAugmentMC, Cutout, RandAugment_no_CutOut 

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

class Client(PSTrainer):

    def __init__(self, client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                 test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num):
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                         test_data_num, device, args, model_trainer)
        if args.VAE == True and vae_model is not None:
            logging.info(f"client {self.client_index} VAE Moel set up")
            self.vae_model = vae_model

        self.test_dataloader = test_dataloader
        self.train_ori_data = train_ori_data  
        self.train_ori_targets = train_ori_targets
        self.train_cls_counts_dict = train_cls_counts_dict
        self.dataset_num = dataset_num

        self.local_num_iterations = math.ceil(len(self.train_ori_data) / self.args.batch_size)

# -------------------------VAE optimization tool for different client------------------------#
        self.vae_optimizer =  AdamW([
            {'params': self.vae_model.parameters()}
        ], lr=1.e-3, betas=(0.9, 0.999), weight_decay=1.e-6)
        self._construct_train_ori_dataloader()
        if self.args.VAE_adaptive:
            self._set_local_traindata_property()
            logging.info(self.local_traindata_property)

    def _construct_train_ori_dataloader(self):
        # ---------------------generate local train dataloader for Fed Step--------------------------#
        train_ori_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            train_ori_transform.transforms.append(RandAugmentMC(n=2, m=10))
        train_ori_transform.transforms.append(transforms.ToTensor())
        
        train_ori_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                                transform=train_ori_transform)
        self.local_train_dataloader = torch.utils.data.DataLoader(dataset=train_ori_dataset,
                                                                  batch_size=32, shuffle=True,
                                                                  drop_last=False)

    def _attack(self,size, mean, std):  #
        rand = torch.normal(mean=mean, std=std, size=size).to(self.device)
        return rand

    def _set_local_traindata_property(self):
        class_num = len(self.train_cls_counts_dict)
        clas_counts = [ self.train_cls_counts_dict[key] for key in self.train_cls_counts_dict.keys()]
        max_cls_counts = max(clas_counts)
        if self.local_sample_number < self.dataset_num/self.args.client_num_in_total * 0.2:
            self.local_traindata_property = 1 # 1 means quantity skew is very heavy
        elif self.local_sample_number > self.dataset_num/self.args.client_num_in_total * 0.2 and max_cls_counts > self.local_sample_number * 0.7:
            self.local_traindata_property = 2 # 2 means label skew is very heavy
        else:
            self.local_traindata_property = None


    def test_local_vae(self, round, epoch, mode):
        # set model as testing mode
        self.vae_model.to(self.device)
        self.vae_model.eval()
        # all_l, all_s, all_y, all_z, all_mu, all_logvar = [], [], [], [], [], []
        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        every_class_acc = {i: 0 for i in range(10)}
        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                batch_size = x.size(0)

                _, _, gx, _, _, rx, rx_noise1, rx_noise2 = self.vae_model(x)

                output = self.vae_model.classifier_test(x)

                loss = F.cross_entropy(output, y)
                prec1, class_acc = accuracy(output.data, y)

                n_iter = round * self.args.VAE_local_epoch + epoch * len(self.test_dataloader) + batch_idx
                test_acc_avg.update(prec1.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                log_info('scalar', 'client {index}:{mode}_test_acc_avg'.format(index=self.client_index, mode=mode),
                         test_acc_avg.avg, step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)
                log_info('scalar', 'client {index}:{mode}_test_loss_avg'.format(index=self.client_index, mode=mode),
                         test_loss_avg.avg, step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)

                total_acc_avg += test_acc_avg.avg

                for key in class_acc.keys():
                    every_class_acc[key] += class_acc[key]
            # plot progress

            for key in every_class_acc.keys():
                every_class_acc[key] = every_class_acc[key] / 10
            logging.info("acc based on different label")
            logging.info(every_class_acc)

            total_acc_avg /= len(self.test_dataloader)
            log_info('scalar', 'client {index}:{mode}_test_loss_avg'.format(index=self.client_index, mode=mode),
                     total_acc_avg,step=round,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)

            logging.info("\n| Testing Epoch #%d\t\tTest Acc: %.4f Test Loss: %.4f" % (
                epoch, test_acc_avg.avg, test_loss_avg.avg))
            print("\n| Testing Epoch #%d\t\tTest Avg Acc: %.4f " % (
                epoch, total_acc_avg))

    def aug_classifier_train(self, round, epoch, optimizer, aug_trainloader):
        self.vae_model.train()
        self.vae_model.training = True

        for batch_idx, (x, y) in enumerate(aug_trainloader):
            x, y, y_b, lam, mixup_index = mixup_data(x, y, alpha=self.args.VAE_alpha)
            x, y, y_b = x.to(self.device), y.to(self.device).view(-1, ), y_b.to(self.device).view(-1, )
            # x, y = Variable(x), [Variable(y), Variable(y_b)]
            x, y = x, [y, y_b]
            n_iter = round * self.args.VAE_local_epoch + epoch * len(aug_trainloader) + batch_idx
            optimizer.zero_grad()

            for name, parameter in self.vae_model.named_parameters():
                if 'classifier' not in name:
                    parameter.requires_grad = False
            out = self.vae_model.get_classifier()(x)

            loss = lam * F.cross_entropy(out, y[0]) + (1. - lam) * F.cross_entropy(out, y[1])
            loss.backward()
            optimizer.step()



    def mosaic(self, batch_data):
        s = 16
        yc, xc = 16, 16
        if self.args.dataset =='fmnist':
            c, w, h = 1, 32, 32
        else:
            c, w, h = 3, 32, 32
        aug_data = torch.zeros((self.args.VAE_aug_batch_size, c, w, h))
        CutOut = Cutout(n_holes=1, length=16)
        for k in range(self.args.VAE_aug_batch_size):

            sample = random.sample(range(batch_data.shape[0]), 4)
            img4 = torch.zeros(batch_data[0].shape)

            left = random.randint(0, 16)
            up = random.randint(0, 16)

            for i, index in enumerate(sample):
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                img4[:, x1a:x2a, y1a:y2a] = batch_data[index][:, left:left + 16, up:up + 16]
            img4 = CutOut(img4)
            aug_data[k] = img4
        return aug_data

    def aug_VAE_train(self, round, epoch, optimizer, aug_trainloader):
        self.vae_model.train()
        self.vae_model.training = True
        self.vae_model.requires_grad_(True)

        for batch_idx, (x, y) in enumerate(aug_trainloader):
            n_iter = round * self.args.VAE_local_epoch + epoch  * len(aug_trainloader) + batch_idx
            batch_size = x.size(0)
            if batch_size < 4:
                break
            # using mosaic data train VAE first for get a good initialize
            aug_data = self.mosaic(x).to(self.device)
            optimizer.zero_grad()

            if self.args.VAE_curriculum:
                if epoch < 100:
                    re = 10 * self.args.VAE_re
                elif epoch < 200:
                    re = 5 * self.args.VAE_re
                else:
                    re = self.args.VAE_re
            else:
                re = self.args.VAE_re

            _, _, aug_gx, aug_mu, aug_logvar, _, _, _ = self.vae_model(aug_data)
            aug_l1 = F.mse_loss(aug_gx, aug_data)
            aug_l3 = -0.5 * torch.sum(1 + aug_logvar - aug_mu.pow(2) - aug_logvar.exp())
            aug_l3 /= self.args.VAE_aug_batch_size * 3 * self.args.VAE_z


            aug_loss = re * aug_l1 + self.args.VAE_kl * aug_l3

            aug_loss.backward()
            optimizer.step()


    def train_whole_process(self, round, epoch, optimizer, trainloader):
        self.vae_model.train()
        self.vae_model.training = True

        loss_avg = AverageMeter()
        loss_rec = AverageMeter()
        loss_ce = AverageMeter()
        loss_entropy = AverageMeter()
        loss_kl = AverageMeter()
        top1 = AverageMeter()


        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, optimizer.param_groups[0]['lr']))

        for batch_idx, (x, y) in enumerate(trainloader):
            n_iter = round * self.args.VAE_local_epoch + epoch * len(trainloader) + batch_idx
            x, y = x.to(self.device), y.to(self.device)

            batch_size = x.size(0)

            if self.args.VAE_curriculum:
                if epoch < 10:
                    re = 10 * self.args.VAE_re
                elif epoch < 20:
                    re = 5 * self.args.VAE_re
                else:
                    re = self.args.VAE_re
            else:
                re = self.args.VAE_re

            optimizer.zero_grad()
            out, hi, gx, mu, logvar, rx, rx_noise1, rx_noise2 = self.vae_model(x)

            cross_entropy = F.cross_entropy(out[: batch_size * 2], y.repeat(2))
            x_ce_loss = F.cross_entropy(out[batch_size * 2:], y)
            l1 = F.mse_loss(gx, x)
            l2 = cross_entropy
            l3 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            l3 /= batch_size * 3 * self.args.VAE_z

            if self.args.VAE_adaptive:
                if self.local_traindata_property == 1 :
                    loss = 5 * re * l1 + self.args.VAE_ce * l2 + 0.5 * self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss
                if self.local_traindata_property == 2 :
                    loss = re * l1 + 5 * self.args.VAE_ce * l2 + 5 * self.args.VAE_kl * l3 + 5 * self.args.VAE_x_ce * x_ce_loss
                if self.local_traindata_property == None:
                    loss = re * l1 + self.args.VAE_ce * l2 + self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss
            else: 
                loss = re * l1 + self.args.VAE_ce * l2 + self.args.VAE_kl * l3 + self.args.VAE_x_ce * x_ce_loss
                
            loss.backward()
            optimizer.step()


            prec1, prec5, correct, pred, class_acc = accuracy(out[:batch_size].data, y[:batch_size].data, topk=(1, 5))
            loss_avg.update(loss.data.item(), batch_size)
            loss_rec.update(l1.data.item(), batch_size)
            loss_ce.update(cross_entropy.data.item(), batch_size)
            loss_kl.update(l3.data.item(), batch_size)
            top1.update(prec1.item(), batch_size)

            log_info('scalar', 'client {index}:loss'.format(index=self.client_index),
                     loss_avg.avg,step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)
            log_info('scalar',  'client {index}:acc'.format(index=self.client_index),
                     top1.avg,step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)

            if epoch % 5 == 0:
                if (batch_idx + 1) % 20 == 0:
                    logging.info('\r')
                    logging.info(
                        '| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Loss_rec: %.4f Loss_ce: %.4f Loss_entropy: %.4f Loss_kl: %.4f Acc@1: %.3f%%'
                        % (epoch, self.args.VAE_local_epoch, batch_idx + 1,
                           len(trainloader), loss_avg.avg, loss_rec.avg, loss_ce.avg, loss_entropy.avg,
                           loss_kl.avg, top1.avg))


    def train_vae_model(self,round):
        train_transform = transforms.Compose([])
        aug_vae_transform_train = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            train_transform.transforms.append(transforms.Resize(32))
            aug_vae_transform_train.transforms.append(transforms.Resize(32))
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            train_transform.transforms.append(RandAugmentMC(n=3, m=10))
        train_transform.transforms.append(transforms.ToTensor())

        aug_vae_transform_train.transforms.append(transforms.RandomCrop(32, padding=4))
        aug_vae_transform_train.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            aug_vae_transform_train.transforms.append(RandAugment_no_CutOut(n=2, m=10))
        aug_vae_transform_train.transforms.append(transforms.ToTensor())
        


        train_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets, transform=train_transform)
        aug_vae_dataset = Dataset_Personalize(self.train_ori_data, self.train_ori_targets,
                                              transform=aug_vae_transform_train)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True,
                                                       drop_last=False)
        aug_vae_dataloader = torch.utils.data.DataLoader(dataset=aug_vae_dataset, batch_size=32, shuffle=True,
                                                         drop_last=False)

        logging.info(f"client {self.client_index} is going to train own VAE-model to generate RX GX and RXnoise")

        self.vae_model.to(self.device)
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + self.args.VAE_local_epoch):
            self.aug_classifier_train(round, epoch, self.vae_optimizer, train_dataloader)
            if self.args.VAE_adaptive == True:
                if self.local_traindata_property == 1 or self.local_traindata_property == None:
                    self.aug_VAE_train(round, epoch, self.vae_optimizer, aug_vae_dataloader)
            else:
                self.aug_VAE_train(round, epoch, self.vae_optimizer, aug_vae_dataloader)
            self.train_whole_process(round, epoch, self.vae_optimizer, train_dataloader)
        self.vae_model.cpu()

    def generate_data_by_vae(self):
        data = self.train_ori_data
        targets = self.train_ori_targets
        generate_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            generate_transform.transforms.append(transforms.Resize(32))
        generate_transform.transforms.append(transforms.ToTensor())
        
        generate_dataset = Dataset_Personalize(data, targets, transform=generate_transform)
        generate_dataloader = torch.utils.data.DataLoader(dataset=generate_dataset, batch_size=self.args.VAE_batch_size,
                                                          shuffle=False, drop_last=False)

        self.vae_model.to(self.device)
        self.vae_model.eval()

        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(generate_dataloader):
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                _, _, gx, _, _, rx, rx_noise1, rx_noise2 = self.vae_model(x)

                batch_size = x.size(0)

                if batch_idx == 0:
                    self.local_share_data1 = rx_noise1
                    self.local_share_data2 = rx_noise2
                    self.local_share_data_y = y
                else:
                    self.local_share_data1 = torch.cat((self.local_share_data1, rx_noise1))
                    self.local_share_data2 = torch.cat((self.local_share_data2, rx_noise2))
                    self.local_share_data_y = torch.cat((self.local_share_data_y, y))




    # got the classifier parameter from the whole VAE model
    def get_generate_model_classifer_para(self):
        return deepcopy(self.vae_model.get_classifier().cpu().state_dict())

    # receive data from server
    def receive_global_share_data(self, data1, data2, y):
        '''
        data: Tensor [num, C, H, W] shared by server collected all clients generated by VAE
        y: Tenosr [num, ] label corrospond to data
        '''
        self.global_share_data1 = data1.cpu()
        self.global_share_y = y.cpu()
        self.global_share_data2 = data2.cpu()


    def sample_iid_data_from_share_dataset(self,share_data1,share_data2, share_y, share_data_mode = 1):
        random.seed(random.randint(0,10000))
        if share_data_mode == 1 and share_data1 is None:
            raise RuntimeError("Not get shared data TYPE1")
        if share_data_mode == 2 and share_data2 is None:
            raise RuntimeError("Not get shared data TYPE2")
        smaple_num = self.local_sample_number
        smaple_num_each_cls = smaple_num // self.args.num_classes
        last = smaple_num - smaple_num_each_cls * self.args.num_classes 
        np_y = np.array(share_y.cpu())
        for label in range(self.args.num_classes):
            indexes = list(np.where(np_y == label)[0])
            sample = random.sample(indexes, smaple_num_each_cls)
            if label == 0:
                if share_data_mode == 1:
                    epoch_data = share_data1[sample]
                elif share_data_mode==2:
                    epoch_data = share_data2[sample]
                epoch_label = share_y[sample]
            else:
                if share_data_mode == 1:
                    epoch_data = torch.cat((epoch_data, share_data1[sample]))
                elif share_data_mode ==2:
                    epoch_data = torch.cat((epoch_data, share_data2[sample]))
                epoch_label = torch.cat((epoch_label, share_y[sample]))

        last_sample =  random.sample(range(self.dataset_num), last) 
        if share_data_mode == 1:
            epoch_data = torch.cat((epoch_data, share_data1[last_sample]))
        elif share_data_mode == 2:
            epoch_data = torch.cat((epoch_data, share_data2[last_sample]))
        epoch_label = torch.cat((epoch_label, share_y[last_sample]))

        # statitics
        unq, unq_cnt = np.unique(np.array(epoch_label.cpu()), return_counts=True)  
        epoch_data_cls_counts_dict = {unq[i]: unq_cnt[i] for i in range(len(unq))}

        return epoch_data, epoch_label


    def construct_mix_dataloader(self, share_data1, share_data2, share_y, round):

        # two dataloader inclue shared data from server and local origin dataloader
        train_ori_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            train_ori_transform.transforms.append(transforms.Resize(32))
        train_ori_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_ori_transform.transforms.append(transforms.RandomHorizontalFlip())
        if self.args.dataset not in ['fmnist']:
            train_ori_transform.transforms.append(RandAugmentMC(n=3, m=10))
        train_ori_transform.transforms.append(transforms.ToTensor())
        # train_ori_transform.transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))

        train_share_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #Aug_Cutout(),
        ])
        epoch_data1, epoch_label1 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=1)
        epoch_data2, epoch_label2 = self.sample_iid_data_from_share_dataset(share_data1, share_data2, share_y, share_data_mode=2)

        train_dataset = Dataset_3Types_ImageData(self.train_ori_data, epoch_data1,epoch_data2,
                                                 self.train_ori_targets,epoch_label1,epoch_label2,
                                                 transform=train_ori_transform,
                                                 share_transform=train_share_transform)
        self.local_train_mixed_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                                  batch_size=32, shuffle=True,
                                                                  drop_last=False)


    def get_local_share_data(self, noise_mode):  # noise_mode means get RXnoise2 or RXnoise2
        if self.local_share_data1 is not None and noise_mode == 1:
            return self.local_share_data1, self.local_share_data_y
        elif self.local_share_data2 is not None and noise_mode == 2:
            return self.local_share_data2, self.local_share_data_y
        else:
            raise NotImplementedError

    def check_end_epoch(self):
        return (
                    self.client_timer.local_outer_iter_idx > 0 and self.client_timer.local_outer_iter_idx % self.local_num_iterations == 0)


    def move_vae_to_cpu(self):
        if str(next(self.vae_model.parameters()).device) == 'cpu':
            pass
        else:
            self.vae_model = self.vae_model.to('cpu')


    def move_to_cpu(self):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            pass
        else:
            self.trainer.model = self.trainer.model.to('cpu')
            # optimizer_to(self.trainer.optimizer, 'cpu')

        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, 'cpu')

    def move_to_gpu(self, device):
        if str(next(self.trainer.model.parameters()).device) == 'cpu':
            self.trainer.model = self.trainer.model.to(device)
        else:
            pass

        # logging.info(self.trainer.optimizer.state.values())
        if len(list(self.trainer.optimizer.state.values())) > 0:
            optimizer_to(self.trainer.optimizer, device)

    def lr_schedule(self, num_iterations, warmup_epochs):
        epochs = self.client_timer.local_outer_epoch_idx
        iterations = self.client_timer.local_outer_iter_idx
        if self.args.sched == "no":
            pass
        else:
            if epochs < warmup_epochs:
                self.trainer.warmup_lr_schedule(iterations)
            else:
                # When epoch begins, do lr_schedule.
                if (iterations > 0 and iterations % num_iterations == 0):
                    self.trainer.lr_schedule(epochs)

    def train(self, share_data1, share_data2, share_y,
              round_idx, named_params, params_type='model',
              global_other_params=None, shared_params_for_simulation=None):
        '''
        return:
        @named_params:   all the parameters in model: {parameters_name: parameters_values}
        @params_indexes:  None
        @local_sample_number: the number of traning set in local
        @other_client_params: in FedAvg is {}
        @local_train_tracker_info:
        @local_time_info:  using this by local_time_info['local_time_info'] = {client_index:   , local_comm_round_idx:,   local_outer_epoch_idx:,   ...}
        @shared_params_for_simulation: not using in FedAvg
        '''

        if self.args.instantiate_all:
            self.move_to_gpu(self.device)
        named_params, params_indexes, local_sample_number, other_client_params, \
        shared_params_for_simulation = self.algorithm_on_train(share_data1, share_data2, share_y, round_idx,
                                                               named_params, params_type,
                                                               global_other_params,
                                                               shared_params_for_simulation)
        if self.args.instantiate_all:
            self.move_to_cpu()

        return named_params, params_indexes, local_sample_number, other_client_params, \
                shared_params_for_simulation

    def set_vae_para(self, para_dict):
        self.vae_model.load_state_dict(para_dict)

    def get_vae_para(self):
        return deepcopy(self.vae_model.cpu().state_dict())

    @abstractmethod
    def algorithm_on_train(self, share_data1, share_data2, share_y,round_idx, 
                           named_params, params_type='model',
                           global_other_params=None,
                           shared_params_for_simulation=None):
        named_params, params_indexes, local_sample_number, other_client_params = None, None, None, None
        return named_params, params_indexes, local_sample_number, other_client_params, shared_params_for_simulation








