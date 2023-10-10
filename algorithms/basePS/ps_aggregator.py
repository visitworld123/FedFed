import copy
import logging
import time
from copy import deepcopy

import torch
import numpy as np

from model.FL_VAE import *
from utils.tool import *
from utils.set import *
import torchvision.transforms as transforms
from utils.log_info import log_info
from utils.data_utils import (
    average_named_params,
    check_type
)
from utils.set import *
from utils.tool import log_info


class PSAggregator(object):
    def __init__(self, train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device, args, model_trainer, vae_model):

        self.trainer = model_trainer
        # preparation for global data
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num

        self.train_data_local_num_dict = train_data_local_num_dict
        self.pre_model_parms = self.get_global_model_params()

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.grad_dict = dict()
        self.sample_num_dict = dict()

        # Saving the client_other_params of clients
        self.client_other_params_dict = dict()

        self.vae_model = vae_model

        # this flag_client_model_uploaded_dict flag dict is commonly used by gradient and model params
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num): 
            self.flag_client_model_uploaded_dict[idx] = False

        self.selected_clients = None


    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def get_global_generator(self):
        return self.trainer.get_generator()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)


    def set_grad_params(self, named_grads):
        self.trainer.set_grad_params(named_grads)

    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()

    def get_vae_param(self):
        return deepcopy(self.vae_model.cpu().state_dict())

    def set_vae_param(self, para_dict):
        self.vae_model.load_state_dict(para_dict)
    def save_classifier(self):
        torch.save(self.trainer.model,'classifier_model_client{}_alpha{}_dataset{}.pth'.format(self.args.client_num_in_total,self.args.partition_alpha,self.args.dataset))
    def save_vae_param(self):
        torch.save(self.vae_model,'vae_model_client{}_alpha{}_dataset{}.pth'.format(self.args.client_num_in_total,self.args.partition_alpha,self.args.dataset))

    def get_generate_model_classifer_para(self):
        return deepcopy(self.vae_model.get_classifier().cpu().state_dict())

    def add_local_trained_result(self, index, model_params, model_indexes, sample_num,
                                 client_other_params=None):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.client_other_params_dict[index] = client_other_params
        self.flag_client_model_uploaded_dict[index] = True

    def get_global_model(self):
        return self.trainer.get_model()

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):  # 更新过的话全部重置
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    # got it sample client
    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            # make sure for each comparison, we are selecting the same clients each round
            np.random.seed(round_idx)
            if self.args.client_select == "random":
                num_clients = min(client_num_per_round, client_num_in_total)
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            else:
                raise NotImplementedError

        logging.info("sampling client_indexes = %s" % str(client_indexes))
        self.selected_clients = client_indexes
        return client_indexes

    def test_on_server_for_all_clients(self, epoch, tracker=None, metrics=None):
        logging.info("################test_on_server_for_all_clients : {}".format(epoch))
        avg_acc = self.trainer.test(epoch, self.test_dataloader, self.device)
        return avg_acc

    def test_on_server_for_round(self, round):
        logging.info("################test_on_server_for_all_clients : {}".format(round))
        avg_acc = self.trainer.test_on_server_for_round(round, self.test_dataloader, self.device)
        return avg_acc

    def test_on_server_by_vae(self, round):
        self.vae_model.to(self.device)
        self.vae_model.eval()

        test_acc_avg = AverageMeter()
        test_loss_avg = AverageMeter()

        total_acc_avg = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_dataloader):
                # distribute data to device
                x, y = x.to(self.device), y.to(self.device).view(-1, )
                batch_size = x.size(0)

                out = self.vae_model.classifier_test(x)

                loss = F.cross_entropy(out, y)
                prec1, _ = accuracy(out.data, y)

                n_iter = (round - 1) * len(self.test_dataloader) + batch_idx
                test_acc_avg.update(prec1.data.item(), batch_size)
                test_loss_avg.update(loss.data.item(), batch_size)

                log_info('scalar','VAE_Server_test', test_acc_avg.avg, 
                         step=n_iter, record_tool=self.args.record_tool, 
                        wandb_record=self.args.wandb_record)

                total_acc_avg += test_acc_avg.avg
            total_acc_avg /= len(self.test_dataloader)
            log_info('scalar','VAE_Server_total_acc',total_acc_avg,
                     step=round,record_tool=self.args.record_tool, 
                     wandb_record=self.args.wandb_record)
            logging.info("\n| VAE Phase Server Testing Round #%d\t\tTest Acc: %.4f Test Loss: %.4f" % (
            round, test_acc_avg.avg, test_loss_avg.avg))
            return total_acc_avg

    def get_average_weight_dict(self, sample_num_list):

        average_weights_dict_list, homo_weights_list = \
            self.trainer.averager.get_average_weight(
                sample_num_list)
        return average_weights_dict_list, homo_weights_list

    def aggregate(self):
        '''
        return:
        @averaged_params:
        @global_other_params:
        @shared_params_for_simulation:
        '''
        start_time = time.time()
        model_list = []
        training_num = 0

        global_other_params = {}
        shared_params_for_simulation = {}

        logging.info("Server is averaging model or adding grads!!")
        sample_num_list = []
        client_other_params_list = []
        for idx in self.selected_clients:
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))  # (training_data_num, model)
            sample_num_list.append(self.sample_num_dict[idx])
            if idx in self.client_other_params_dict:
                client_other_params = self.client_other_params_dict[idx]
            else:
                client_other_params = {}
            client_other_params_list.append(client_other_params)
            training_num += self.sample_num_dict[idx]

        logging.debug("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        average_weights_dict_list, _ = self.get_average_weight_dict(
            sample_num_list=sample_num_list)

        averaged_params = average_named_params(
            model_list,  # from sampled client model_list  [(sample_number, model_params)]
            average_weights_dict_list
        )


        # ========================SCAFFOLD=====================#
        if self.args.scaffold:
            c_delta_para_list = []
            for i, client_other_params in enumerate(client_other_params_list):
                c_delta_para_list.append(client_other_params["c_delta_para"])

            total_delta = copy.deepcopy(c_delta_para_list[0])
            # for key, param in total_delta.items():
            #     param.data = 0.0
            for key in total_delta:
                total_delta[key] = 0.0

            for c_delta_para in c_delta_para_list:
                for key, param in total_delta.items():
                    total_delta[key] += c_delta_para[key] / len(client_other_params_list)

            c_global_para = self.c_model_global.state_dict()
            for key in c_global_para:
                # logging.debug(f"total_delta[key].device : {total_delta[key].device}, \
                # c_global_para[key].device : {c_global_para[key].device}")

                c_global_para[key] += check_type(total_delta[key], c_global_para[key].type())
            self.c_model_global.load_state_dict(c_global_para)
            global_other_params["c_model_global"] = c_global_para
        # ========================SCAFFOLD=====================#

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params, global_other_params, shared_params_for_simulation



    def server_generate_data_by_vae(self):
        generate_transform = transforms.Compose([])
        if self.args.dataset == 'fmnist':
            generate_transform.transforms.append(transforms.Resize(32))
        generate_transform.transforms.append(transforms.ToTensor())
        generate_dataset = torchvision.datasets.CIFAR10(self.args.data_dir, train=True, download=True, transform=generate_transform)

        # generate_transform.transforms.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))

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
                    self.global_share_dataset1 = rx_noise1
                    self.global_share_dataset2 = rx_noise2
                    self.global_share_data_y = y
                else:
                    self.global_share_dataset1 = torch.cat((self.global_share_dataset1, rx_noise1))
                    self.global_share_dataset2 = torch.cat((self.global_share_dataset2, rx_noise2))
                    self.global_share_data_y = torch.cat((self.global_share_data_y, y))


        generate_reconst_images(999, self.global_share_dataset1, 'RXnoise1',
                                batch_idx,record_tool=self.args.record_tool, wandb_record=self.args.wandb_record)
        generate_reconst_images(999, self.global_share_dataset1, 'RXnoise2',
                                batch_idx,record_tool=self.args.record_tool, wandb_record=self.args.wandb_record)


