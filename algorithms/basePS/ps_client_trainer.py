import logging

class PSTrainer(object):

    def __init__(self, client_index, train_ori_data, train_ori_targets,test_dataloader, train_data_num,
                test_data_num, device, args, model_trainer):

        self.args = args
        self.client_index = client_index
        self.train_ori_data = train_ori_data
        self.train_ori_targets = train_ori_targets
        self.test_dataloader = test_dataloader
        self.local_sample_number = train_data_num
        self.test_data_num = test_data_num



        logging.info(f"Initializing client: {self.client_index}"+
                    f" len(train_data) (local data num): {len(self.train_ori_data)} ")


        self.device = device
        self.trainer = model_trainer
        # =============================================

    def update_state(self, **kwargs):
        self.trainer.update_state(**kwargs)



    def lr_schedule(self, progress):
        self.trainer.lr_schedule(progress)

    def warmup_lr_schedule(self, iterations):
        self.trainer.warmup_lr_schedule(iterations)


    def set_model_params(self, weights):
        self.trainer.set_model_params(weights)

    def set_grad_params(self, named_grads):
        self.trainer.set_grad_params(named_grads)

    def clear_grad_params(self):
        self.trainer.clear_grad_params()

    def update_model_with_grad(self):
        self.trainer.update_model_with_grad()

    def get_train_batch_data(self):
        try:
            train_batch_data = self.train_local_iter.next()
            logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
            if len(train_batch_data[0]) < self.args.batch_size:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), self.args.batch_size))

        except:
            self.train_local_iter = iter(self.train_local)
            train_batch_data = self.train_local_iter.next()
        return train_batch_data


    def get_model_params(self):
        weights = self.trainer.get_model_params()
        model_indexes = None

        return weights, model_indexes


