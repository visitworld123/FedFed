import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()





""" filter layers """

def filter_parameters(named_parameters=None, named_modules=None,
                    param_layers_list=[""],
                    param_layers_length=-1,
                    param_types=["Conv2d","Linear"]
                    ):

    filtered_named_parameters = {}
    filtered_parameters_crt_names = []
    filtered_parameter_shapes = {}

    if len(param_layers_list) > 0:
        filtered_parameters_crt_names = param_layers_list
    else:
        for name, param in named_parameters.items():
            module_name = '.'.join(name.split('.')[:-1])
            if type(named_modules[module_name]).__name__ in param_types:
                filtered_parameters_crt_names.append(name)

        if param_layers_length == -1:
            pass
        else:
            filtered_parameters_crt_names = filtered_parameters_crt_names[:param_layers_length]

        for name in filtered_parameters_crt_names:
            filtered_named_parameters[name] = {}
            filtered_parameter_shapes[name] = named_parameters[name].shape
    return filtered_parameters_crt_names, filtered_named_parameters, filtered_parameter_shapes


""" scan model with depth """
def scan_model_with_depth(model, param_types=[]):

    named_parameters = dict(model.named_parameters())
    named_modules = dict(model.named_modules())

    layers_depth = {}
    current_depth = 0

    for name, param in named_parameters.items():
        module_name = '.'.join(name.split('.')[:-1])
        if (len(param_types) > 0 and type(named_modules[module_name]).__name__ in param_types) or \
            len(param_types) == 0:
            if module_name not in layers_depth:
                current_depth += 1
                layers_depth[module_name] = current_depth
        else:
            pass
        # params_group.append({'params': model.classifier.parameters(), 'lr': 1e-3})
        # params_group.append({'params': param, "layer_name": "module_name", "depth": layers_depth[module_name]})

    return named_parameters, named_modules, layers_depth


def scan_model_dict_with_depth(model, model_state_dict, param_types=[]):
    named_modules = dict(model.named_modules())

    layers_depth = {}
    current_depth = 0

    for name, param in model_state_dict.items():
        module_name = '.'.join(name.split('.')[:-1])
        if (len(param_types) > 0 and type(named_modules[module_name]).__name__ in param_types) or \
            len(param_types) == 0:
            if module_name not in layers_depth:
                current_depth += 1
                layers_depth[module_name] = current_depth
        else:
            pass
        # params_group.append({'params': model.classifier.parameters(), 'lr': 1e-3})
        # params_group.append({'params': param, "layer_name": "module_name", "depth": layers_depth[module_name]})

    return named_modules, layers_depth




def mean_std_online_estimate(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    # count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def retrieve_mean_std(existingAggregate):
    (count, mean, M2) = existingAggregate
    # logging.info(f"type(count): {type(count)},\
    #     type(mean): {type(mean)},\
    #     type(M2): {type(M2)}")
    if count < 2:
        logging.info(f"count: {count} is less than 2, Error of estimating STD....")
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)



"""some aggregation functions."""


def get_params(model, args):
    """
        some features maybe needed in this
    """
    params = [
        {
            "params": [value],
            "name": key,
            "weight_decay": args.wd if "bn" not in key else 0.0,
            "param_size": value.size(),
            "nelement": value.nelement(),
        }
        for key, value in model.named_parameters()
    ]
    return params


def _get_data(param_groups, idx, is_get_grad):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    else:
        return param_groups[idx]["params"][0]


def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True):
    data, shapes = [], []
    for idx, _ in param_names:
        _data = _get_data(param_groups, idx, is_get_grad)
        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes


def get_named_data(model, mode='MODEL', use_cuda=True):
    """
        getting the whole model and getting the gradients can be conducted
        by using different methods for reducing the communication.
        `model` choices: ['MODEL', 'GRAD', 'MODEL+GRAD'] 
    """
    if mode == 'MODEL':
        own_state = model.cpu().state_dict()
        return own_state
    elif mode == 'GRAD':
        grad_of_params = {}
        for name, parameter in model.named_parameters():
                
            if use_cuda:
                grad_of_params[name] = parameter.grad
            else:
                grad_of_params[name] = parameter.grad.cpu()
          
        return grad_of_params
    elif mode == 'MODEL+GRAD':
        model_and_grad = {}
        for name, parameter in model.named_parameters():

            if use_cuda:
                model_and_grad[name] = parameter.data
                model_and_grad[name+b'_gradient'] = parameter.grad
            else:
                model_and_grad[name] = parameter.data.cpu()
                model_and_grad[name+b'_gradient'] = parameter.grad.cpu()

        return model_and_grad 

# 一个Batch_Normalization的信息
def get_bn_params(prefix, module, use_cuda=True):
    bn_params = {}
    if use_cuda:
        bn_params[f"{prefix}.weight"] = module.weight
        bn_params[f"{prefix}.bias"] = module.bias
        bn_params[f"{prefix}.running_mean"] = module.running_mean
        bn_params[f"{prefix}.running_var"] = module.running_var
        bn_params[f"{prefix}.num_batches_tracked"] = module.num_batches_tracked
    else:
        bn_params[f"{prefix}.weight"] = module.weight.cpu()
        bn_params[f"{prefix}.bias"] = module.bias.cpu()
        bn_params[f"{prefix}.running_mean"] = module.running_mean.cpu()
        bn_params[f"{prefix}.running_var"] = module.running_var.cpu()
        bn_params[f"{prefix}.num_batches_tracked"] = module.num_batches_tracked
    return bn_params

# 找到所有Batch_Normalization
def get_all_bn_params(model, use_cuda=True):
    all_bn_params = {}
    for module_name, module in model.named_modules():

        if type(module) is nn.BatchNorm2d:

            bn_params = get_bn_params(module_name, module, use_cuda=use_cuda)
            all_bn_params.update(bn_params)
    return all_bn_params


def idv_average_named_params(named_params_list, average_weights_dict_list, homo_weights_list=[],
        inplace=True):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Different for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))

    if inplace:
        (_, averaged_params) = named_params_list[0]
    else:
        (_, averaged_params) = deepcopy(named_params_list[0])

    for name in averaged_params.keys():
        for i in range(0, len(named_params_list)):
            local_sample_number, local_named_params = named_params_list[i]
            # logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
            #     local_sample_number, sum))
            # The w here could be a tensor or a scaler.
            if name not in average_weights_dict_list[i]:
                # logging.info(f"Warnning: Layer {name} not in average_weights_dict_list[{i}].\
                #     Using homo_weights_list[{i}] as averaging weights instead...")
                w = homo_weights_list[i]
            else:
                w = average_weights_dict_list[i][name]
                logging.debug(f"average_weights_dict_list[{i}][{name}]: {average_weights_dict_list[i][name]}")
                # w = torch.full_like(average_weights_dict_list[i][name], homo_weights_list[i]).detach()
                # logging.debug(f"homo_weights_list[i]: {homo_weights_list[i]}")
            if not isinstance(w, float):
                local_named_params[name] = check_device(local_named_params[name], w.device)
            if i == 0:
                averaged_params[name] = (local_named_params[name].data * w \
                    ).type(averaged_params[name].dtype)
            else:
                averaged_params[name] = check_device(averaged_params[name], local_named_params[name].data.device)
                averaged_params[name] += (local_named_params[name].data * w \
                    ).type(averaged_params[name].dtype)
    return averaged_params


def average_named_params(named_params_list, average_weights_dict_list):
    """
        This is a weighted average operation.
        average_weights_dict_list: includes weights with respect to clients. Same for each param.
        inplace:  Whether change the first client's model inplace.
    """
    # logging.info("################aggregate: %d" % len(named_params_list))

    if type(named_params_list[0]) is tuple or type(named_params_list[0]) is list:
        (_, averaged_params) = deepcopy(named_params_list[0])
    else:
        averaged_params = deepcopy(named_params_list[0])
#   averaged_params is the dict of all parameters      #
    for k in averaged_params.keys():
        for i in range(0, len(named_params_list)):  # model个数
            if type(named_params_list[0]) is tuple or type(named_params_list[0]) is list:
                local_sample_number, local_named_params = named_params_list[i]
            else:
                local_named_params = named_params_list[i]
            # logging.debug("aggregating ---- local_sample_number/sum: {}/{}, ".format(
            #     local_sample_number, sum))
            w = average_weights_dict_list[i]  # 不同client的权重
            # w = torch.full_like(local_named_params[k], w).detach()
            if i == 0:
                averaged_params[k] = (local_named_params[k] * w).type(averaged_params[k].dtype)
            else:
                averaged_params[k] += (local_named_params[k].to(averaged_params[k].device) * w).type(
                    averaged_params[k].dtype)
    return averaged_params


def average_tensors(tensors, weights, inplace=False):
    if isinstance(tensors, list) or isinstance(tensors, np.ndarray):
        sum = np.sum(weights)
        if inplace:
            averaged_tensor = tensors[0]
        else:
            averaged_tensor = deepcopy(tensors[0])
        for i, tensor in enumerate(tensors):
            w = weights[i] / sum
            if i == 0:
                averaged_tensor = tensor.to(averaged_tensor.device) * w
            else:
                averaged_tensor += tensor.to(averaged_tensor.device) * w
    elif isinstance(tensors, dict):
        # logging.info(f"type(weights): {type(weights)}")
        sum = np.sum(list(weights.values()))
        averaged_tensor = None
        for i, key in enumerate(tensors.keys()):
            w = weights[key] / sum
            if i == 0:
                averaged_tensor = tensors[key].to(averaged_tensor.device) * w
            else:
                averaged_tensor += tensors[key].to(averaged_tensor.device) * w
    else:
        raise NotImplementedError()
    return averaged_tensor


"""tensor reshape."""

def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1], dtype=tensors[0].dtype,
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec


def unflatten(tensors, synced_tensors, shapes):
    pointer = 0

    for tensor, shape in zip(tensors, shapes):
        param_size, nelement = shape
        tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement


def flatten_model(named_parameters=None, flatten_grad=False, param_list=None):
    
    to_concat_w = []
    to_concat_g = []
    # if named_parameters is None:
    #     named_parameters = model.named_parameters()

    for name, param in named_parameters.items():
        # TODO maybe custom the layers that need to be pruned.
        if param.dim() in [2, 4]:
            to_concat_w.append(param.data.view(-1))
            if flatten_grad:
                to_concat_g.append(param.grad.data.view(-1))

    all_w = torch.cat(to_concat_w)
    if flatten_grad:
        all_g = torch.cat(to_concat_g)
    else:
        all_g = None
    return all_w, all_g 






"""auxiliary."""


def recover_device(data, device=None):
    if device is not None:
        return data.to(device)
    else:
        return data


def check_device(data_src, device=None):
    if device is not None:
        if data_src.device is not device:
            return data_src.to(device)
        else:
            return data_src
    else:
        return data_src


def check_type(data_src, type=None):
    if type is not None:
        if data_src.type() == type:
            return data_src
        else:
            return data_src.type(type)
    else:
        return data_src


def deepcopy_model(conf, model):
    # a dirty hack....
    tmp_model = deepcopy(model)
    if conf.track_model_aggregation:
        for tmp_para, para in zip(tmp_model.parameters(), model.parameters()):
            tmp_para.grad = para.grad.clone()
    return tmp_model


def get_name_params_div(named_parameters1, named_parameters2=None, scalar=1.0):
    """
        return named_parameters2 - named_parameters1
    """
    if named_parameters2 is not None:
        common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
        named_diff_parameters = {}
        for key in common_names:
            named_diff_parameters[key] = get_div_weights(named_parameters1[key],
                                            weights2=named_parameters2[key])
    else:
        named_diff_parameters = {}
        for key in named_parameters1.keys():
            named_diff_parameters[key] = get_div_weights(named_parameters1[key],
                                            scalar=scalar)
    return named_diff_parameters


def get_name_params_sum(named_parameters1, named_parameters2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    named_diff_parameters = {}
    for key in common_names:
        named_diff_parameters[key] = get_sum_weights(named_parameters1[key], named_parameters2[key])
    return named_diff_parameters


def get_name_params_difference(named_parameters1, named_parameters2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    named_diff_parameters = {}
    for key in common_names:
        named_diff_parameters[key] = get_diff_weights(named_parameters1[key], named_parameters2[key])
    return named_diff_parameters


def get_name_params_difference_abs(named_parameters1, named_parameters2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    named_diff_parameters = {}
    for key in common_names:
        named_diff_parameters[key] = get_diff_weights_abs(named_parameters1[key], named_parameters2[key])
    return named_diff_parameters


def get_name_params_difference_norm(named_parameters1, named_parameters2, layers_list=None, p=2):
    """
        return named_parameters2 - named_parameters1
    """
    common_names = list(set(named_parameters1.keys()).intersection(set(named_parameters2.keys())))
    name_params_difference_norm = {}
    for name in common_names:
        if layers_list is None:
            weight_diff = get_diff_weights(named_parameters1[name], named_parameters2[name])
            # logging.info(f'name: {name}, weight_diff: {weight_diff}')
            name_params_difference_norm[name] = weight_diff.norm(p=p)
        else:
            for layer in layers_list:
                if layer in name:
                    weight_diff = get_diff_weights(named_parameters1[name], named_parameters2[name])
                    name_params_difference_norm[name] = weight_diff.norm(p=p)
    return name_params_difference_norm



def get_tensors_norm(tensors_dict, layers_list=None, p=2):
    tensors_norm_dict = {}

    try:
        for name, tensor in tensors_dict.items():
            if layers_list is None:
                if tensor is not None:
                    tensors_norm_dict[name] = tensor.norm(p=p)
                else:
                    tensors_norm_dict[name] = None
            else:
                for layer in layers_list:
                    # logging.info("layer in list: {}, get {} in tensor dict".format(
                    #     layer, name))
                    if layer in name:
                        if tensor is not None:
                            tensors_norm_dict[name] = tensor.norm(p=p)
                        else:
                            tensors_norm_dict[name] = None
    except:
        logging.info("layers_list: {}, Layer name: {}, tensor.shape: {}, tensor.dtype: {}".format(
            layers_list, name, tensor.shape, tensor.dtype
        ))
    return tensors_norm_dict




def get_div_weights(weights1, weights2=None, scalar=1.0):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if weights2 is not None:
        if isinstance(weights1, list) and isinstance(weights2, list):
            return [torch.div(w1, w2) for (w1, w2) in zip(weights1, weights2)]
        elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
            return torch.div(weights1, weights2)
        else:
            raise NotImplementedError
    else:
        if isinstance(weights1, list):
            return [w1 / scalar for w1 in weights1]
        elif isinstance(weights1, torch.Tensor):
            return  weights1 / scalar
        else:
            raise NotImplementedError



def get_sum_weights(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [w2 + w1 for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return weights2 + weights1
    else:
        raise NotImplementedError

def get_diff_weights(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [w2 - w1 for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return weights2 - weights1
    else:
        raise NotImplementedError


def get_diff_weights_abs(weights1, weights2):
    """ Produce a direction from 'weights1' to 'weights2'."""
    if isinstance(weights1, list) and isinstance(weights2, list):
        return [torch.abs(w2 - w1) for (w1, w2) in zip(weights1, weights2)]
    elif isinstance(weights1, torch.Tensor) and isinstance(weights2, torch.Tensor):
        return torch.abs(weights2 - weights1)
    else:
        raise NotImplementedError


def get_diff_tensor_norm(tensor1, tensor2, p=2):
    diff = get_diff_weights(tensor1, tensor2)
    return diff.norm(p=p)


def get_diff_tensor_norm_with_dimnorm(tensor1, tensor2, p=2):
    diff = get_diff_weights(tensor1, tensor2)
    tensor1_norm = tensor1.norm(p=p)
    tensor2_norm = tensor2.norm(p=p)
    dim = diff.numel()
    diff_norm = diff.norm(p=p)
    return diff_norm * (1/(dim ** (1/p))), diff_norm / (0.5*tensor1_norm + 0.5*tensor2_norm)



def get_diff_tensor_norm_with_originnorm(tensor1, tensor2, p=2):
    diff = get_diff_weights(tensor1, tensor2)
    tensor1_norm = tensor1.norm(p=p)
    tensor2_norm = tensor2.norm(p=p)
    return diff.norm(p=p) / (0.5*tensor1_norm + 0.5*tensor2_norm)




def get_diff_states(states1, states2):
    """ Produce a direction from 'states1' to 'states2'."""
    return [
        v2 - v1
        for (k1, v1), (k2, v2) in zip(states1.items(), states2.items())
    ]


def get_tensor_rotation(tensor1, tensor2):
    logging.info(tensor1.dtype)
    logging.info(tensor2.dtype)
    # print(tensor1.dtype)
    # print(tensor2.dtype)
    # return F.cosine_similarity(torch.flatten(tensor1), torch.flatten(tensor2), dim=0)
    return F.cosine_similarity(tensor1.view(-1), tensor2.view(-1), dim=0)


def get_named_tensors_rotation(named_tensors1, named_tensors2, layers_list=None):
    common_names = list(set(named_tensors1.keys()).intersection(set(named_tensors2.keys())))
    named_tensors_rotation = {}
    for name in common_names:
        if layers_list is None:
            # print(name)
            named_tensors_rotation[name] = get_tensor_rotation(named_tensors1[name], named_tensors2[name])
        else:
            for layer in layers_list:
                if layer in name:
                    # print(name)
                    named_tensors_rotation[name] = get_tensor_rotation(named_tensors1[name], named_tensors2[name])
    return named_tensors_rotation


def add_gaussian_noise_named_tensors(named_tensor, mean=0.0, std=0.001):
    for name, _ in named_tensor.items():
        noise = torch.normal(mean=torch.ones(named_tensor[name].shape)*mean, std=std)
        named_tensor[name] += noise.to(named_tensor[name].device)
    return named_tensor


def calculate_metric_for_tensor(
    cal_func,
    tensor1, tensor2=None, LP_list=None):
    if LP_list is None:
        metric_for_named_tensors = cal_func(
            tensor1,
            tensor2,
        )
        return metric_for_named_tensors
    else:
        assert type(LP_list) is list
        metric_for_named_tensors_with_LP_dict = {}

        for p in LP_list:
            if p == 'inf':
                Lp = float('inf')
            else:
                Lp = float(p)
            metric_for_named_tensors = cal_func(
                tensor1,
                tensor2,
                p=Lp
            )
            metric_for_named_tensors_with_LP_dict[p] = metric_for_named_tensors
    return metric_for_named_tensors_with_LP_dict





def calculate_metric_for_named_tensors(
    cal_func,
    named_tensors1, named_tensors2=None, layers_list=None, LP_list=None):

    if LP_list is None:
        if named_tensors2 is not None:
            metric_for_named_tensors = cal_func(
                named_tensors1,
                named_tensors2,
                layers_list=layers_list,
            )
        else:
            metric_for_named_tensors = cal_func(
                named_tensors1,
                layers_list=layers_list,
            )
        return metric_for_named_tensors
    else:
        assert type(LP_list) is list
        metric_for_named_tensors_with_LP_dict = {}

        for p in LP_list:
            if p == 'inf':
                Lp = float('inf')
            else:
                Lp = float(p)
            if named_tensors2 is not None:
                metric_for_named_tensors = cal_func(
                    named_tensors1,
                    named_tensors2,
                    layers_list=layers_list,
                    p=Lp
                )
            else:
                metric_for_named_tensors = cal_func(
                    named_tensors1,
                    layers_list=layers_list,
                    p=Lp
                )
            metric_for_named_tensors_with_LP_dict[p] = metric_for_named_tensors
        return metric_for_named_tensors_with_LP_dict


def calculate_metric_for_whole_model(
    cal_func,
    named_tensors1, named_tensors2, layers_list=None, LP_list=None):

    common_names = list(set(named_tensors1.keys()).intersection(set(named_tensors2.keys())))
    list_of_tensors1 = []
    list_of_tensors2 = []

    for name in common_names:
        if layers_list is None:
            # print(name)
            list_of_tensors1.append(named_tensors1[name]) 
            list_of_tensors2.append(named_tensors2[name]) 
        else:
            for layer in layers_list:
                if layer in name:
                    list_of_tensors1.append(named_tensors1[name]) 
                    list_of_tensors2.append(named_tensors2[name]) 

    merged_tensor1 = list_to_vec(list_of_tensors1)
    merged_tensor2 = list_to_vec(list_of_tensors2)

    if LP_list is None:
        metric_for_named_tensors = cal_func(
            merged_tensor1,
            merged_tensor2,
        )
        return metric_for_named_tensors
    else:
        assert type(LP_list) is list
        metric_for_named_tensors_with_LP_dict = {}

        for p in LP_list:
            if p == 'inf':
                Lp = float('inf')
            else:
                Lp = float(p)
            metric_for_named_tensors = cal_func(
                merged_tensor1,
                merged_tensor2,
                p=Lp
            )
            metric_for_named_tensors_with_LP_dict[p] = metric_for_named_tensors
        return metric_for_named_tensors_with_LP_dict



def calculate_metric_for_layers(
    cal_func,
    layer_params1, layer_params2, layers_list=None, LP_list=None):

    common_names = list(set(layer_params1.keys()).intersection(set(layer_params2.keys())))
    metric_for_named_tensors = {}
    for name in common_names:
        list_of_layer_params1 = []
        list_of_layer_params2 = []
        for i, _ in enumerate(layer_params1[name]):
            list_of_layer_params1.append(layer_params1[name][i]['params']) 
            list_of_layer_params2.append(layer_params2[name][i]['params'])

        merged_tensor1 = list_to_vec(list_of_layer_params1)
        merged_tensor2 = list_to_vec(list_of_layer_params2)

        if layers_list is None:
            metric_for_tensor_with_LP_dict = calculate_metric_for_tensor(
                cal_func, tensor1=merged_tensor1, tensor2=merged_tensor2, LP_list=None)
            metric_for_named_tensors[name] = metric_for_tensor_with_LP_dict
        else:
            for layer in layers_list:
                if layer in name:
                    metric_for_tensor_with_LP_dict = calculate_metric_for_tensor(
                        cal_func, tensor1=merged_tensor1, tensor2=merged_tensor2, LP_list=None)
                    metric_for_named_tensors[name] = metric_for_tensor_with_LP_dict
    return metric_for_named_tensors



def calc_client_layer_divergence(global_layer_params, layer_params_list, p=2, rotation=False):
    layer_divergence_dimnorm_list = {}
    layer_divergence_originnorm_list = {}

    for key, _ in global_layer_params.items():
        layer_divergence_dimnorm_list[key] = []
        layer_divergence_originnorm_list[key] = []

    for i, layer_params in enumerate(layer_params_list):
        metric_for_named_tensors = calculate_metric_for_layers(
            cal_func=get_diff_tensor_norm_with_dimnorm,
            layer_params1=global_layer_params,
            layer_params2=layer_params,
            LP_list=[p],
        )
        # logging.info(f"metric_for_named_tensors: {metric_for_named_tensors}")
        # for key in [metric_for_named_tensors.keys()]:
        for key, value in metric_for_named_tensors.items():
            # if key not in [layer_divergence_list.keys()]:
            #     layer_divergence_list[key] = []
            # layer_divergence_list[key].append(metric_for_named_tensors[key][p].item())
            # logging.info(f"metric_for_named_tensors[{key}]: {metric_for_named_tensors[key]}")
            layer_divergence_dimnorm_list[key].append(metric_for_named_tensors[key][0].item())
            layer_divergence_originnorm_list[key].append(metric_for_named_tensors[key][1].item())

    layer_average_divergence_dimnorm = {}
    layer_average_divergence_originnorm = {}

    for key, _ in layer_divergence_dimnorm_list.items():
        divergence_dimnorm_list = layer_divergence_dimnorm_list[key]
        divergence_originnorm_list = layer_divergence_originnorm_list[key]

        average_divergence_dimnorm = sum(divergence_dimnorm_list) / len(divergence_dimnorm_list)
        layer_average_divergence_dimnorm[key] = average_divergence_dimnorm
        average_divergence_originnorm = sum(divergence_originnorm_list) / len(divergence_originnorm_list)
        layer_average_divergence_originnorm[key] = average_divergence_originnorm

    return layer_divergence_dimnorm_list, layer_average_divergence_dimnorm, \
        layer_divergence_originnorm_list, layer_average_divergence_originnorm





def calc_client_divergence(global_model_weights, model_list, p=2, rotation=False):
    divergence_list = []
    if rotation:
        for i, model_i in enumerate(model_list):
            model_diff_norm1 = calculate_metric_for_whole_model(
                cal_func=get_tensor_rotation,
                named_tensors1=global_model_weights,
                named_tensors2=model_list[i][1],
                LP_list=None,
            )
            divergence_list.append(model_diff_norm1.item())
        average_divergence = sum(divergence_list) / len(divergence_list)
        max_divergence = max(divergence_list)
        min_divergence = min(divergence_list)
    else:
        for i, model_i in enumerate(model_list):
            model_diff_norm1 = calculate_metric_for_whole_model(
                cal_func=get_diff_tensor_norm,
                named_tensors1=global_model_weights,
                named_tensors2=model_list[i][1],
                LP_list=[p],
            )
            divergence_list.append(model_diff_norm1[p].item())
        average_divergence = sum(divergence_list) / len(divergence_list)
        max_divergence = max(divergence_list)
        min_divergence = min(divergence_list)

    return divergence_list, average_divergence, max_divergence, min_divergence



def get_model_difference(model1, model2, p=2):
    list_of_tensors = []
    for weight1, weight2 in zip(model1.parameters(),
                                model2.parameters()):
        tensor = get_diff_weights(weight1, weight2)
        list_of_tensors.append(tensor)
    return list_to_vec(list_of_tensors).norm(p=p).item()


def list_to_vec(weights):
    """ Concatnate a numpy list of weights of all layers into one torch vector.
    """
    v = []
    direction = [d * np.float64(1.0) for d in weights]
    for w in direction:
        if isinstance(w, np.ndarray):
            w = torch.tensor(w)
        else:
            w = w.clone().detach()
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def is_float(value):
    try:
        float(value)
        return True
    except:
        return False



"""gradient related"""
# TODO
def apply_gradient(param_groups, state, apply_grad_to_model=True):
    """
        SGD
    """
    for group in param_groups:
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        dampening = group["dampening"]
        nesterov = group["nesterov"]

        for p in group["params"]:
            if p.grad is None:
                continue
            d_p = p.grad.data

            # get param_state
            param_state = state[p]

            # add weight decay.
            if weight_decay != 0:
                d_p.add_(p.data, alpha=weight_decay)

            # apply the momentum.
            if momentum != 0:
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
            if apply_grad_to_model:
                p.data.add_(d_p, alpha=-group["lr"])
            else:
                p.grad.data = d_p




def clear_grad(m):
    for p in m.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()




"""dataset related"""

def get_local_num_iterations(local_num, batch_size):
    return local_num // batch_size


def get_min_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the minimum iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    min_local_num = 10000000
    for worker_idx, local_num in train_data_local_num_dict.items():
        if min_local_num > local_num:
            min_local_num = local_num
    return min_local_num // batch_size


def get_max_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the maximum iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    max_local_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        if max_local_num < local_num:
            max_local_num = local_num
    return max_local_num // batch_size


# got it 平均需要的iterations数 = 总数据量/work数量/batch_size=每个work平均数据量/batch_size
def get_avg_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the averaged iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    sum_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        sum_num += local_num
    num_workers = len(train_data_local_num_dict.keys())
    return (sum_num // num_workers) // batch_size


def get_sum_num_iterations(train_data_local_num_dict, batch_size):
    """
        This is used to get the averaged iteration of all clients.
        Note: For APSGD and SSPSGD, this function is different,
            because local client records their local epochs.
    """
    sum_num = 0
    for worker_idx, local_num in train_data_local_num_dict.items():
        sum_num += local_num
    return sum_num // batch_size

#  got it
def get_num_iterations(train_data_local_num_dict, batch_size, type="default", default=10):
    """
        [default, lowest, highest, averaged]  
    """
    if type == "default":
        num_iterations = default
    elif type == "lowest":
        num_iterations = get_min_num_iterations(train_data_local_num_dict, batch_size)
    elif type == "highest":
        num_iterations = get_max_num_iterations(train_data_local_num_dict, batch_size)
    elif type == "averaged":
        num_iterations = get_avg_num_iterations(train_data_local_num_dict, batch_size)
    else:
        raise NotImplementedError
    return num_iterations



def get_train_batch_data(train_local_iter_dict, dataset_name, train_local, batch_size, drop_last=True):
    try:
        train_batch_data = train_local_iter_dict[dataset_name].next()
        # logging.debug("len(train_batch_data[0]): {}".format(len(train_batch_data[0])))
        if len(train_batch_data[0]) < batch_size:
            if drop_last:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), batch_size))
                logging.debug("Using Drop Last, reinitialize loader.")
                train_local_iter_dict[dataset_name] = iter(train_local)
                train_batch_data = train_local_iter_dict[dataset_name].next()
            else:
                logging.debug("WARNING: len(train_batch_data[0]): {} < self.args.batch_size: {}".format(
                    len(train_batch_data[0]), batch_size))

            # logging.debug("train_batch_data[0]: {}".format(train_batch_data[0]))
            # logging.debug("train_batch_data[0].shape: {}".format(train_batch_data[0].shape))
    except:
        train_local_iter_dict[dataset_name] = iter(train_local)
        train_batch_data = train_local_iter_dict[dataset_name].next()
    return train_batch_data



""" data distribution """
def get_num_cls_in_batch(batch_data, cls_idx):
    return len(batch_data[batch_data == cls_idx])


def get_label_distribution(train_data_local_dict, class_num):
    local_cls_num_list_dict = {}
    total_cls_num = {}
    for label in range(class_num):
        total_cls_num[label] = 0
    for client in train_data_local_dict.keys():
        logging.info("In get_label_distribution: travelling client: {} ".format(client))
        local_cls_num_list_dict[client] = [0 for _ in range(class_num)]
        for _, labels in train_data_local_dict[client]:
            for cls_idx in range(class_num):
                num_cls = get_num_cls_in_batch(labels, cls_idx)
                local_cls_num_list_dict[client][cls_idx] += num_cls
                total_cls_num[cls_idx] += num_cls
# 返回字典形返回不同client中不同类别的个数，保存在local_cls_num_list_dict，不同类别的个数，保存在total_cls_num
# local_cls_num_list_dict = {client_idx:[labe0_num_client_idx,...,label9_num_client_idx]}
    return local_cls_num_list_dict, total_cls_num




def get_selected_clients_label_distribution(local_cls_num_list_dict, class_num, client_indexes, min_limit=0):
    logging.info(local_cls_num_list_dict)
    selected_clients_label_distribution = [0 for _ in range(class_num)]
    for client_index in client_indexes:
        # selected_train_data_local_num_dict[client_index] = [0 for _ in range(class_num)]
        for cls_idx in range(class_num):
            selected_clients_label_distribution[cls_idx] += local_cls_num_list_dict[client_index][cls_idx]
    if min_limit > 0:
        for i in range(class_num):
            if selected_clients_label_distribution[i] < min_limit:
                selected_clients_label_distribution[i] = min_limit
    return selected_clients_label_distribution


def get_per_cls_weights(cls_num_list, beta=0.9999):
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    # per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)



""" cpu --- gpu """
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)










