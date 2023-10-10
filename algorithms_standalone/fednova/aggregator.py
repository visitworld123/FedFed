
from algorithms_standalone.basePS.aggregator import Aggregator


class FedNovaAggregator(Aggregator):
    def __init__(self,train_dataloader, test_dataloader, train_data_num, test_data_num,
                 train_data_local_num_dict, worker_num, device,args, model_trainer,vae_model):
        super().__init__(train_dataloader, test_dataloader, train_data_num, test_data_num,
                        train_data_local_num_dict, worker_num, device,args, model_trainer, vae_model)








