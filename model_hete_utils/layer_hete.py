import argparse
import torch
from models.resnet import Resnet
from models.vit import SViT
from models.vgg import Staged_VGG
from spikingjelly.activation_based import layer, neuron, functional
import copy
from collections import OrderedDict
from torch import nn
import math
import numpy as np
import torch.nn.functional as F


# layer hete is in fact stage-layer hete
# stage-hete hete
# layer-hete hete
# stage-layer-hete

# for vgg and resnet usage
# inputs = torch.load("../cifar10-sample.pth", map_location="cpu")
# inputs = inputs.repeat(4, 1, 1, 1, 1)
# inputs = inputs.to("cuda:0")


class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes, reduction=1, scale=1., is_snn=True):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.reduction = reduction
        self.scale = scale

        self.fc = layer.Linear(in_planes, num_classes)
        functional.set_step_mode(self, "m")

    def forward(self, inp, pred=None):
        output = F.adaptive_avg_pool3d(inp, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.backbone = nn.ModuleList()

    def init(self, layers):
        for l in layers:
            self.backbone.append(copy.deepcopy(l))
        # self.backbone = nn.Sequential(*self.backbone)

    def forward(self, x):
        for n in self.backbone:
            x = n(x)
        return x


class ee_model(nn.Module):
    def __init__(self, main_ee_model, cls):
        super(ee_model, self).__init__()
        self.backbone = main_ee_model
        self.cls = cls

    def forward(self, x):
        x = self.backbone(x)
        return self.cls(x)


class layer_hete:
    def __init__(self, args):
        self.num_hete_models = args.num_hete_models
        # this hete is very very hard to deal with when handle resnet,vgg,vit is ok
        # train it normally by introducing a light-weight ee-classifier
        self.ee_layer_locations = args.ee_layer_locations  # []
        self.ee_layer_planes = args.ee_layer_planes
        # every local model has a light-weight ee-classifier
        # self.ee_level = [0,1,2]
        # portion
        self.args = args
        self.ee_classifiers = [Classifier(in_planes=e_plane,
                                          num_classes=args.num_classes) for e_plane in self.ee_layer_planes]

    def distribute(self, global_model, sampled_users, global_epoch):
        if "vgg" in self.args:
            self.ee_classifiers = [copy.deepcopy(global_model.fc) for _ in self.ee_layer_planes]
        hete_ids = []
        local_parameters = [OrderedDict() for _ in range(self.args.n_parties)]
        client_models = []
        if "resnet" in self.args.model:
            all_blocks = nn.ModuleList(
                [global_model.conv1, *global_model.layer1, *global_model.layer2, *global_model.layer3,
                 *global_model.layer4])
        elif "vit" in self.args.model:
            all_blocks = nn.ModuleList([global_model.patch_embed, *global_model.block])
        else:
            all_blocks = nn.ModuleList([*global_model.layer0, *global_model.layer1, *global_model.layer2,
                                       *global_model.layer3,*global_model.layer4])
        for i, num in enumerate(self.args.portion):
            num = math.ceil(self.args.n_parties * num)
            for _ in range(num):
                hete_ids.append(i)
        for net_i in range(self.args.n_parties):
            ee_level = hete_ids[net_i]
            # if ee_level == hete_ids[-1]:  # global model on local device
            #     client_model = type(global_model)(self.args)
            # else:
            ee_location = self.ee_layer_locations[ee_level]
            backbone = BackBone()
            backbone.init(all_blocks[:ee_location + 1])
            classifier = self.ee_classifiers[ee_level]
            client_model = ee_model(backbone, classifier)
            client_models.append(client_model)
            local_parameters[net_i] = client_model.state_dict()
        return client_models, local_parameters
        pass

    def aggregate(self, global_model, local_parameters, sampled_users):
        # default to be fedavg
        count = OrderedDict()
        update_state_dict = copy.deepcopy(global_model.state_dict())
        for k, v in update_state_dict.items():
            count[k] = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
            for u_id in sampled_users:
                local_parameter = local_parameters[u_id]
                for lk, lv in local_parameter.items():
                    if "backbone" in lk:
                        # ee model
                        new_lk = parse_ee_weight_key(lk, self.args.model, global_model.num_block)
                        if new_lk == k:
                            count[new_lk] += 1
                            tmp_v += lv
                    elif "cls" in lk:
                        # cls
                        # adopt fc weight of global model on local device first
                        # we have more methods here
                        # kd, momentum..
                        if lv.shape == v.shape:
                            tmp_v += lv
                            count[k] += 1
                        pass
                    else:
                        # global model
                        if lk == k:
                            count[k] += 1
                            tmp_v += lv
                    pass
            tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        global_model.load_state_dict(update_state_dict)
        pass


def parse_ee_weight_key(k, model_name, stage_blocks=None):
    new_k = None
    suffix = k.split(".")[2:]
    layer_id = int(suffix[0])
    suffix.pop(0)
    suffix = ".".join(suffix)
    if "resnet" in model_name:
        # stage_blocks = [3,4,6,3]
        temp_array = np.zeros((sum(stage_blocks)), dtype=int)
        cum_array = np.cumsum(stage_blocks)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i - 1]:] += 1
        if layer_id == 0:
            new_k = "conv1." + suffix
        else:
            layer_id -= 1
            stage_id = temp_array[layer_id]
            if stage_id == 0:
                block_id = layer_id
            else:
                block_id = layer_id - np.cumsum(stage_blocks[:stage_id])[-1]
                block_id = 0 if block_id < 0 else block_id
            new_k = f"layer{stage_id + 1}.{block_id}.{suffix}"
    else:
        stage_blocks = [2,2,3,3,3]
        temp_array = np.zeros((sum(stage_blocks)), dtype=int)
        cum_array = np.cumsum(stage_blocks)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i - 1]:] += 1
        else:
            layer_id -= 1
            stage_id = temp_array[layer_id]
            if stage_id == 0:
                block_id = layer_id
            else:
                block_id = layer_id - np.cumsum(stage_blocks[:stage_id])[-1]
                block_id = 0 if block_id < 0 else block_id
            new_k = f"layer{stage_id + 1}.{block_id}.{suffix}"
    return new_k








    # feature_signal = feature_signal.flatten()
    # print(feature_signal)
    # print(output)
    # distributor = layer_hete(args)
    # client_models,_ = distributor.distribute(global_model,[],0)

    # input.requires_grad = True
    # energy_count(client_models[2].backbone,(input,))
    # stage_blocks = [[1,1,1,1],[2,2,2,2],[3,4,6,3]]
    # keys = {}
    # for i,c in enumerate(client_models):
    #     keys[i] = []
    #     for k in c.state_dict().keys():
    #         if "cls" in k:
    #             continue
    #         keys[i].append(parse_ee_weight_key(k,"resnet",[3,4,6,3]))
    #     print(keys[i])
    pass