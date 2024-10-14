import math
from collections import OrderedDict
import copy
import torch
import time

class stage_hete:
    def __init__(self,args):
        self.num_hete_models = args.num_hete_models
        # this is much simple, realize stage split by simply regulate local models
        self.args = args
        self.model_hete = args.model_hete

    def distribute(self,global_model,sampled_users,global_epoch):
        client_models = {}
        hete_ids = []
        for i,num in enumerate(self.args.portion):
            num = math.ceil(self.args.n_parties * num)
            for _ in range(num):
                hete_ids.append(i)
        local_parameters = {net_id: OrderedDict() for net_id in sampled_users}
        global_state_dict = global_model.state_dict()
        for net_i in sampled_users:
            # randomly choose a hete model from 3
            hete_id = hete_ids[net_i]
            hidden_blocks = self.model_hete  # [4,6,8]
            self.args.num_hidden_layers = self.model_hete
            if "resnet" in self.args.model:
                model_name = self.args.model[:-2] + str(hidden_blocks[hete_id] * 2 + 2)
            else:
                model_name = self.args.model[:3] + str(hidden_blocks[hete_id])
            self.args.model = model_name
            client_model = type(global_model)(self.args)
            client_models[net_i] = client_model
            for k in client_model.state_dict().keys():
                local_parameters[net_i][k] = copy.deepcopy(global_state_dict[k])  # directly copy, this is resnet, vgg needs more concern
        # for resnet,vgg,it has multiple stages
        # direct select corresponding weight from global model
        return client_models,local_parameters,hete_ids
        pass

    def aggregate(self,global_model,local_parameters,sampled_users):
        # default to be fedavg
        # st = time.time()
        updated_state_dict = copy.deepcopy(global_model.state_dict())
        count = OrderedDict()
        for k,v in updated_state_dict.items():
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for net_id in sampled_users:
                local_parameter = local_parameters[net_id]
                for lk,lv in local_parameter.items():
                    if lk == k:
                        count[k] += 1
                        tmp_v += lv
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        # et = time.time()
        # print("agg time consum:",et-st)
        global_model.load_state_dict(updated_state_dict)
        pass