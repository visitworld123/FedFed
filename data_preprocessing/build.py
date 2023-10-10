import os
import logging




from .loader import Data_Loader







# got it
def load_data(load_as, args=None, process_id=0, mode="standalone", task="federated", data_efficient_load=True,
                dirichlet_balance=False, dirichlet_min_p=None,
                dataset="", datadir="./", partition_method="hetero", partition_alpha=0.1, client_number=1, batch_size=128, num_workers=4,
                data_sampler=None,
                resize=32, augmentation="default"):
    '''
    return:
    @train_date_num: means the number of training data from dataset totally, and usually the server training data number
    @test_data_num: means the number of testing data totally from dataset totally, and usually the server test data number
    @train_data_global: the DataLoader for Server and in this case is the whole dataset's training set
    @test_data_global: the DataLoader for Server and in this case is the whole dataset's testing set
    @data_local_num_dict: training set number of each clinet: {clinet_index: num_training_data}
    @train_data_dataloader_local_dict: Dict of DataLoader for each client: {clinet_index: Train_DataLoader_cliet_id}
    @test_data_dataloader_local_dict: Dict of DataLoader for each client: {clinet_index: Test_DataLoader_cliet_id}
    @class_num: total class number of the dataset
    @other_params: Dict for other parameters
    '''

    # datadir = get_new_datadir(args, datadir, dataset)
    other_params = {}

    data_loader = Data_Loader(args, process_id, mode, task, data_efficient_load, dirichlet_balance, dirichlet_min_p,
        dataset, datadir, partition_method, partition_alpha, client_number, batch_size, num_workers,
        data_sampler,
        resize=resize, augmentation=augmentation, other_params=other_params)
    train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl, train_data_local_num_dict, \
    test_data_local_num_dict, test_data_local_dl_dict, train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
    other_params = data_loader.load_data()  # FL mode load data

    return train_data_global_num, test_data_global_num, train_data_global_dl, test_data_global_dl,train_data_local_num_dict, \
        test_data_local_num_dict, test_data_local_dl_dict, train_data_local_ori_dict, train_targets_local_ori_dict, class_num, \
        other_params











def load_multiple_centralized_dataset(load_as, args, process_id, mode, task,
                        dataset_list, datadir_list, batch_size, num_workers,
                        data_sampler=None,
                        resize=32, augmentation="default"): 
    train_dl_dict = {}
    test_dl_dict = {}
    train_ds_dict = {}
    test_ds_dict = {}
    class_num_dict = {}
    train_data_num_dict = {}
    test_data_num_dict = {}

    for i, dataset in enumerate(dataset_list):
        # kwargs["data_dir"] = datadir_list[i]
        datadir = datadir_list[i]
        # train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
        #     = load_centralized_data(load_as, args, process_id, mode, task,
        #                 dataset, datadir, batch_size, num_workers,
        #                 data_sampler=None,
        #                 resize=resize, augmentation=augmentation)
        train_dl, test_dl, train_data_num, test_data_num, class_num, other_params \
            = load_data(load_as=load_as, args=args, process_id=process_id,
                        mode="centralized", task="centralized",
                        dataset=dataset, datadir=datadir, batch_size=args.batch_size, num_workers=args.data_load_num_workers,
                        data_sampler=None,
                        resize=resize, augmentation=augmentation)

        train_dl_dict[dataset] = train_dl
        test_dl_dict[dataset] = test_dl
        train_ds_dict[dataset] = other_params["train_ds"]
        test_ds_dict[dataset] = other_params["test_ds"]
        class_num_dict[dataset] = class_num
        train_data_num_dict[dataset] = train_data_num
        test_data_num_dict[dataset] = test_data_num

    return train_dl_dict, test_dl_dict, train_ds_dict, test_ds_dict, \
        class_num_dict, train_data_num_dict, test_data_num_dict












