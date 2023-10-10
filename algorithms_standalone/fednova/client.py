import logging
import copy

import torch

from algorithms_standalone.basePS.client import Client


class FedNovaClient(Client):
    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num, perf_timer=None, metrics=None):
        super().__init__(client_index, train_ori_data, train_ori_targets, test_dataloader, train_data_num,
                test_data_num,  train_cls_counts_dict, device, args, model_trainer, vae_model, dataset_num, perf_timer, metrics)
        local_num_iterations_dict = {}
        local_num_iterations_dict[self.client_index] = self.local_num_iterations

        self.global_epochs_per_round = self.args.global_epochs_per_round
        local_num_epochs_per_comm_round_dict = {}
        local_num_epochs_per_comm_round_dict[self.client_index] = self.args.global_epochs_per_round

    # override
    def lr_schedule(self, num_iterations, warmup_epochs):
        epoch = None
        iteration = None
        round_idx = self.client_timer.local_comm_round_idx
        if self.args.sched == "no":
            pass
        else:
            if round_idx < warmup_epochs:
                # Because gradual warmup need iterations updates
                self.trainer.warmup_lr_schedule(round_idx*num_iterations)
            else:
                self.trainer.lr_schedule(round_idx)


    def fednova_train(self, share_data1, share_data2, share_y,round_idx=None, shared_params_for_simulation=None):
        previous_model = copy.deepcopy(self.trainer.get_model_params())
        client_other_params = {}
        self.move_to_gpu(self.device)

        tau = 0
        for epoch in range(self.args.global_epochs_per_round):
            self.construct_mix_dataloader(share_data1, share_data2, share_y,round_idx)
            self.trainer.train_mix_dataloader(epoch, self.local_train_mixed_dataloader, self.device)
            tau = len(self.local_train_mixed_dataloader)
            logging.info("#############train finish for {epoch}  epoch and test result on client {index} ########".format(
                    epoch=epoch, index=self.client_index))

        a_i = (tau - self.args.momentum * (1 - pow(self.args.momentum, tau)) / (1 - self.args.momentum)) / (1 - self.args.momentum)
        global_model_para = previous_model
        net_para = self.trainer.get_model_params()
        norm_grad = copy.deepcopy(previous_model)
        for key in norm_grad:

            norm_grad[key] = torch.true_divide(global_model_para[key]-net_para[key], a_i)



        self.move_to_cpu()
        client_other_params["a_i"] = a_i
        client_other_params["norm_grad"] = norm_grad
        # return None, None, self.local_sample_number, a_i, norm_grad
        return None, None, self.local_sample_number, client_other_params, shared_params_for_simulation


    def algorithm_on_train(self, update_state_kargs,
            client_index, named_params, params_type='model', traininig_start=False,
            shared_params_for_simulation=None):
        pass
















