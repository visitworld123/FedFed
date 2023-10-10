import copy
import logging

import torch

from .client import FedNovaClient
from .aggregator import FedNovaAggregator
from utils.data_utils import get_avg_num_iterations
from model.FL_VAE import *
from algorithms_standalone.basePS.basePSmanager import BasePSManager
from model.build import create_model
from trainers.build import create_trainer


class FedNovaManager(BasePSManager):
    def __init__(self, device, args):
        super().__init__(device, args)

        self.global_epochs_per_round = self.args.global_epochs_per_round
     

    def _setup_server(self):
        logging.info("############_setup_server (START)#############")
        model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                             device=self.device, **self.other_params)
        init_state_kargs = {}  
        if self.args.VAE:
            VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

        model_trainer = create_trainer( 
            self.args, self.device, model, train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num, train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl, train_data_local_num_dict=self.train_data_local_num_dict,
            class_num=self.class_num, server_index=0, role='server', **init_state_kargs
        )

        # model_trainer = create_trainer(self.args, self.device, model)
        self.aggregator = FedNovaAggregator(self.train_data_global_dl, self.test_data_global_dl,
                                           self.train_data_global_num,
                                           self.test_data_global_num, self.train_data_local_num_dict,
                                           self.args.client_num_in_total, self.device,
                                           self.args, model_trainer, VAE_model)


        logging.info("############_setup_server (END)#############")

    def _setup_clients(self):
        logging.info("############setup_clients (START)#############")
        init_state_kargs = self.get_init_state_kargs()  
        # for client_index in range(self.args.client_num_in_total):
        for client_index in range(self.number_instantiated_client):
            if self.args.VAE:
                VAE_model = FL_CVAE_cifar(args=self.args, d=self.args.VAE_d, z=self.args.VAE_z, device=self.device)

            model = create_model(self.args, model_name=self.args.model, output_dim=self.args.model_output_dim,
                                 device=self.device, **self.other_params)

            num_iterations = get_avg_num_iterations(self.train_data_local_num_dict,
                                                    self.args.batch_size)  
            model_trainer = create_trainer(self.args, self.device, model, class_num=self.class_num,
                                           other_params=self.other_params, client_index=client_index, role='client',
                                           **init_state_kargs)

            client = FedNovaClient(client_index, train_ori_data=self.train_data_local_ori_dict[client_index],
                                  train_ori_targets=self.train_targets_local_ori_dict[client_index],
                                  test_dataloader=self.test_data_local_dl_dict[client_index],
                                  train_data_num=self.train_data_local_num_dict[client_index],
                                  test_data_num=self.test_data_local_num_dict[client_index],
                                  train_cls_counts_dict=self.train_cls_local_counts_dict[client_index],
                                  device=self.device, args=self.args, model_trainer=model_trainer,
                                  vae_model=VAE_model, dataset_num=self.train_data_global_num,
                                  perf_timer=self.perf_timer, metrics=self.metrics)
            # client.train_vae_model()
            self.client_list.append(client)
        logging.info("############setup_clients (END)#############")

    # override
    def check_end_epoch(self):
        return True



    def train(self):
        for round in range(self.comm_round):

            logging.debug("################Communication round : {}".format(self.server_timer.global_comm_round_idx))
            # w_locals = []

            global_model_params = self.aggregator.get_global_model_params()


            client_indexes = self.aggregator.client_sampling(
                round, self.args.client_num_in_total,
                self.args.client_num_per_round)
            logging.debug("client_indexes = " + str(client_indexes))

            a_list = {}
            d_list = {}
            n_list = {}

            global_time_info = self.server_timer.get_time_info_to_send()
            update_state_kargs = self.get_update_state_kargs()
            for i, client_index in enumerate(client_indexes):
                if self.args.instantiate_all:
                    client = self.client_list[client_index]
                else:
                    client = self.client_list[i]

                if self.args.exchange_model:
                    copy_global_model_params = copy.deepcopy(global_model_params)
                    client.set_model_params(copy_global_model_params) 
                client.move_to_gpu(self.device)


                global_other_params = {}
                shared_params_for_simulation = {}

                model_params, model_indexes, local_sample_number, client_other_params, shared_params_for_simulation = \
                    client.fednova_train( self.global_share_dataset1, self.global_share_dataset2, self.global_share_data_y,
                                          round_idx=client.client_timer.global_comm_round_idx,
                                        shared_params_for_simulation=shared_params_for_simulation)
                a_i, d_i = client_other_params["a_i"], client_other_params["norm_grad"]

                client.move_to_cpu()
                a_list[client_index] = a_i
                d_list[client_index] = d_i
                n_i = local_sample_number
                n_list[client_index] = n_i

            total_n = sum(n_list.values())
            d_total_round = copy.deepcopy(global_model_params)
            for key in d_total_round:
                d_total_round[key] = 0.0

            for client_index in client_indexes:
                d_para = d_list[client_index]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[client_index] / total_n

            # update global model
            coeff = 0.0
            for client_index in client_indexes:
                coeff = coeff + a_list[client_index] * n_list[client_index] / total_n

            # global_model_params = global_model.state_dict()
            for key in global_model_params:
                #print(updated_model[key])
                if global_model_params[key].type() == 'torch.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif global_model_params[key].type() == 'torch.cuda.LongTensor':
                    global_model_params[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    #print(updated_model[key].type())
                    #print((coeff*d_total_round[key].type()))
                    global_model_params[key] -= coeff * d_total_round[key]
            self.aggregator.set_global_model_params(global_model_params)
            # -----------------test model on server every communication round------------------#
            avg_acc = self.aggregator.test_on_server_for_round(self.args.VAE_comm_round + round)
            self.test_acc_list.append(avg_acc)
            print(avg_acc)
            if round % 20 == 0:
                print(self.test_acc_list)


    def algorithm_train(self, client_indexes, named_params, params_type,
                        update_state_kargs, global_time_info, shared_params_for_simulation):
        pass




