import copy




class Averager(object):
    """
        Responsible to implement average.
        There maybe some history information need to be memorized.
    """
    def __init__(self, args, model):
        self.args = args

    def get_average_weight(self, sample_num_list, avg_weight_type='datanum'):
        # balance_sample_number_list = []
        average_weights_dict_list = []
        sum = 0
        inv_sum = 0 

        sample_num_list = copy.deepcopy(sample_num_list)
        # for i in range(0, len(sample_num_list)):
        #     sample_num_list[i] * np.random.random(1)

        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]
            inv_sum = None
            sum += local_sample_number

        for i in range(0, len(sample_num_list)):
            local_sample_number = sample_num_list[i]

            if avg_weight_type == 'datanum':
                weight_by_sample_num = local_sample_number / sum

            average_weights_dict_list.append(weight_by_sample_num)

        homo_weights_list = average_weights_dict_list
        return average_weights_dict_list, homo_weights_list











