import logging
import numpy as np

import torch
from torch import nn

from utils.data_utils import (
    get_named_data,
    get_all_bn_params,
    check_device
)
from trainers.averager import Averager
from utils.set import *
from utils.log_info import *
from utils.tool import *

class NormalTrainer(object):
    def __init__(self, model, device, criterion, optimizer, lr_scheduler, args, **kwargs):
        


        if kwargs['role'] == 'server':
            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index
            self.client_index = None
            self.index = self.server_index

        elif kwargs['role'] == 'client':
            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index
            self.server_index = None
            self.index = self.client_index
        else:
            raise NotImplementedError

        self.role = kwargs['role']

        self.args = args
        self.model = model

        self.device = device
        self.criterion = criterion.to(device)
        self.optimizer = optimizer


        self.param_groups = self.optimizer.param_groups  

        self.named_parameters = list(self.model.named_parameters())  # tuple [(name,param),(),...,()]
        if len(self.named_parameters) > 0:
            self._parameter_names = {v: k for k, v
                                    in sorted(self.named_parameters)}
            #print('Sorted named_parameters')
        else:
            self._parameter_names = {v: 'noname.%s' % i
                                    for param_group in self.param_groups
                                    for i, v in enumerate(param_group['params'])}

        self.averager = Averager(self.args, self.model) # it doesn't matter

        self.lr_scheduler = lr_scheduler

    def epoch_init(self):
        pass

    def epoch_end(self):
        pass

    def track(self, tracker, summary_n_samples, model, loss, end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):
        pass

    def update_state(self, **kwargs):
        # This should be called begin the training of each epoch.
        self.update_loss_state(**kwargs)


    def get_model_named_modules(self):
        return dict(self.model.cpu().named_modules())


    def get_model(self):
        return self.model


    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        # for name, param in model_parameters.items():
        #     logging.info(f"Getting params as model_parameters: name:{name}, shape: {param.shape}")
        self.model.load_state_dict(model_parameters)



    def set_feature_align_means(self, feature_align_means):
        self.feature_align_means = feature_align_means
        self.align_feature_loss.feature_align_means = feature_align_means

    def get_feature_align_means(self):
        return self.feature_align_means

    def get_model_bn(self):
        all_bn_params = get_all_bn_params(self.model)
        return all_bn_params

# got it Batch_Normalization set
    def set_model_bn(self, all_bn_params):
        # logging.info(f"all_bn_params.keys(): {all_bn_params.keys()}")
        # for name, params in all_bn_params.items():
            # logging.info(f"name:{name}, params.shape: {params.shape}")
        for module_name, module in self.model.named_modules():
            if type(module) is nn.BatchNorm2d:
                # logging.info(f"module_name:{module_name}, params.norm: {module.weight.data.norm()}")
                module.weight.data = all_bn_params[module_name+".weight"] 
                module.bias.data = all_bn_params[module_name+".bias"] 
                module.running_mean = all_bn_params[module_name+".running_mean"] 
                module.running_var = all_bn_params[module_name+".running_var"] 
                module.num_batches_tracked = all_bn_params[module_name+".num_batches_tracked"] 

#  `mode` choices: ['MODEL', 'GRAD', 'MODEL+GRAD']
    def get_model_grads(self):
        named_mode_data = get_named_data(self.model, mode='GRAD', use_cuda=True)
        # logging.info(f"Getting grads as named_grads: {named_grads}")
        return named_mode_data

    def set_grad_params(self, named_grads):
        # pass
        self.model.train()
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            parameter.grad.copy_(named_grads[name].data.to(self.device)) # 把改name的grad复制过来


    def clear_grad_params(self):
        self.optimizer.zero_grad()

    def update_model_with_grad(self):
        self.model.to(self.device)
        self.optimizer.step()

    def get_optim_state(self):
        return self.optimizer.state


    def clear_optim_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()


    def lr_schedule(self, progress):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(progress)
        else:
            logging.info("No lr scheduler...........")


    def warmup_lr_schedule(self, iterations):
        if self.lr_scheduler is not None:
            self.lr_scheduler.warmup_step(iterations)



    def get_train_batch_data(self, train_dataloader):
        try:
            train_batch_data = self.train_local_iter.next()
            # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))
                # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
                # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
        except:
            self.train_local_iter = iter(train_dataloader)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data



    def train_mix_dataloader(self, epoch, trainloader, device, **kwargs):
        self.model.to(device)
        self.model.train()
        self.model.training =True

        loss_avg = AverageMeter()
        acc = AverageMeter()

        logging.info('\n=> Training Epoch #%d, LR=%.4f' % (epoch, self.optimizer.param_groups[0]['lr']))
        for batch_idx, (x1, x2,x3, y1, y2,y3) in enumerate(trainloader):
            x1, x2, x3, y1, y2,y3 = x1.to(device), x2.to(device), x3.to(device), \
                                    y1.to(device), y2.to(device), y3.to(device)

            batch_size = x1.size(0)
            self.optimizer.zero_grad()

            x = torch.cat((x1, x2,x3))
            y = torch.cat((y1,y2,y3))

            out = self.model(x)

            loss = self.criterion(out, y)

            # ========================FedProx=====================#
            if self.args.fedprox:
                fed_prox_reg = 0.0
                previous_model = kwargs["previous_model"]
                for name, param in self.model.named_parameters():
                    fed_prox_reg += ((self.args.fedprox_mu / 2) * \
                        torch.norm((param - previous_model[name].data.to(device)))**2)
                loss += fed_prox_reg
            # ========================FedProx=====================#

            loss.backward()
            self.optimizer.step()

            # ========================SCAFFOLD=====================#
            if self.args.scaffold:
                c_model_global = kwargs['c_model_global']
                c_model_local = kwargs['c_model_local']
                if self.lr_scheduler is not None:
                    current_lr = self.lr_scheduler.lr
                else:
                    current_lr = self.args.lr
                for name, param in self.model.named_parameters():
                    # logging.debug(f"c_model_global[name].device : {c_model_global[name].device}, \
                    #     c_model_local[name].device : {c_model_local[name].device}")
                    param.data = param.data - 0.000001 * \
                                 check_device((c_model_global[name] - c_model_local[name]), param.data.device)
            # ========================SCAFFOLD=====================#

            prec1, prec5, correct, pred, _ = accuracy(out.data, y.data, topk=(1, 5))

            loss_avg.update(loss.data.item(), batch_size)
            acc.update(prec1.data.item(), batch_size)

            n_iter = (epoch - 1) * len(trainloader) + batch_idx

            log_info('scalar', '{role}_{index}_train_loss_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     loss_avg.avg,step=n_iter,record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)
            log_info('scalar', '{role}_{index}_train_acc_epoch {epoch}'.format(role=self.role, index=self.index, epoch=epoch),
                     acc.avg,step=n_iter,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)




    def test_on_server_for_round(self, round, testloader, device):
        self.model.to(device)
        self.model.eval()

        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        total_loss_avg = 0
        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                # distribute data to device
                x, y = x.to(device), y.to(device).view(-1, )
                batch_size = x.size(0)

                out = self.model(x)

                loss = self.criterion(out, y)
                prec1, _ = accuracy(out.data, y)

                n_iter = (round - 1) * len(testloader) + batch_idx
                test_acc_avg.update(prec1.data.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                log_info('scalar', '{role}_{index}_test_acc_epoch'.format(role=self.role, index=self.index),
                         test_acc_avg.avg, step=n_iter,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
                total_loss_avg += test_loss_avg.avg
                total_acc_avg += test_acc_avg.avg
            total_acc_avg /= len(testloader)
            total_loss_avg /= len(testloader)
            log_info('scalar', '{role}_{index}_total_acc_epoch'.format(role=self.role, index=self.index),
                     total_acc_avg, step=round,record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
            log_info('scalar', '{role}_{index}_total_loss_epoch'.format(role=self.role, index=self.index),
                     total_loss_avg, step=round, record_tool=self.args.record_tool,
                     wandb_record=self.args.wandb_record)
            return total_acc_avg
