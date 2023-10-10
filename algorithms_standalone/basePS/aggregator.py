import os
import sys

from algorithms.basePS.ps_aggregator import PSAggregator

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

class Aggregator(PSAggregator):
    def __init__(self, train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model):
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model)


    def get_max_comm_round(self):
        pass






















