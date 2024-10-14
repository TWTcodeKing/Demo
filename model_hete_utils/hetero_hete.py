import numpy as np
import torch
from model_hete_utils.hetero_fl_utils import split_model
from collections import OrderedDict
import copy


class hetero_hete:
    def __init__(self,args):
        self.num_hete_models = args.num_hete_models
        self.model_rate = args.mode_rate
        self.proportion = (np.array(args.proportion) / sum(args.proportion)).tolist()
        self.rate_idx = torch.multinomial(torch.tensor(self.proportion), num_samples=args.n_parties,
                                     replacement=True).tolist()
        self.model_rate = np.array(self.model_rate)[self.rate_idx]
        self.args = args
        self.tmp_counts = {}
        self.param_idx = None
        classes_count = np.multiply(args.n_parties, list(reversed(self.proportion))).astype(np.int32)
        self.rate_idx = []
        for idx,num in enumerate(classes_count):
            for _ in range(num):
                self.rate_idx.append(idx)
    def distribute(self,global_model,sampled_users,global_epoch):
        client_models = {}
        for i in range(self.args.n_parties):
            self.args.rate = self.model_rate[i]
            client_model = type(global_model)(self.args)
            client_model.train()
            client_models[i] = client_model
        for k, v in global_model.state_dict().items():
            self.tmp_counts[k] = torch.ones_like(v)
        global_parameters = global_model.state_dict()
        param_idx = split_model(global_parameters,self.model_rate,sampled_users,global_epoch,self.args)
        # no idea about this
        local_parameters = [OrderedDict() for _ in range(self.args.n_parties)]
        for k, v in global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(sampled_users)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if 'weight' in parameter_type:
                        if v.dim() > 1:
                            local_parameters[sampled_users[m]][k] = copy.deepcopy(v[torch.meshgrid(param_idx[sampled_users[m]][k])])
                        else:
                            local_parameters[sampled_users[m]][k] = copy.deepcopy(v[param_idx[sampled_users[m]][k]])
                    else:
                        local_parameters[sampled_users[m]][k] = copy.deepcopy(v[param_idx[sampled_users[m]][k]])
                elif "running" in parameter_type:
                    local_parameters[sampled_users[m]][k] = copy.deepcopy(v[param_idx[sampled_users[m]][k]])
                else:
                    local_parameters[sampled_users[m]][k] = copy.deepcopy(v)
        self.param_idx = param_idx
        return client_models, local_parameters, self.rate_idx

    def aggregate(self,global_model,local_parameters,sampled_users):
        count = OrderedDict()
        updated_parameters = copy.deepcopy(global_model.state_dict())
        param_idx = self.param_idx
        if "resnet" in self.args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                # if 'fc' in k:
                                #     label_split = all_label_split[sampled_users[m]]
                                #     param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                #     param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                #     tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                #     count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                # else:
                                tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            # if 'fc' in k:
                            #     label_split = all_label_split[sampled_users[m]]
                            #     param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                            #     tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                            #     count[k][param_idx[sampled_users[m]][k]] += 1
                            # else:
                            tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                            count[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += local_parameters[sampled_users[m]][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif "vit" in self.args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                # if "patch_embed" in k:
                                #     label_split = all_label_split[sampled_users[m]]
                                #     param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                #     param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                #     tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                #     count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                # elif 'head' in k:
                                #     label_split = all_label_split[sampled_users[sampled_users[m]]]
                                #     param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                #     param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                #     tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                #     count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                # else:
                                tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            # if 'head' in k:
                            #     label_split = all_label_split[sampled_users[m]]
                            #     param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                            #     tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                            #     count[k][param_idx[sampled_users[m]][k]] += 1
                            # else:
                            tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                            count[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += local_parameters[sampled_users[m]][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif "vgg" in self.args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                # if 'fc' in k:
                                #     label_split = all_label_split[sampled_users[m]]
                                #     param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                #     param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                #     tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                #     count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                # else:
                                tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            # if 'fc' in k:
                            #     label_split = all_label_split[sampled_users[m]]
                            #     param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                            #     tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                            #     count[k][param_idx[sampled_users[m]][k]] += 1
                            # else:
                            tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                            count[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += local_parameters[sampled_users[m]][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        global_model.load_state_dict(updated_parameters)
