import copy

import numpy as np

from model_hete_utils import *
from model_hete_utils.in_fl_utils import process_model_grad,Aggregator_mom,HeteAgg_mom
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch import optim
from collections import OrderedDict
from spikingjelly.activation_based import functional
from utils import Logger,compute_accuracy,stratified_sampling_indices
import time
from models.resnet import Resnet
from models.vgg import Staged_VGG
from models.vit import SViT
import re
import math
import torch.nn.functional as F
# normal fedavg

class fedavg_server:
    def __init__(self,global_model,datasets,distributor,args):
        self.device = torch.device(args.device)
        self.global_model = global_model
        self.global_epochs = args.global_epochs
        self.local_epochs = args.local_epochs
        self.hete = distributor
        self.local_models,self.local_parameters = None, None
        self.dist_tr_dataset = datasets['train']
        self.dist_ts = datasets['test']
        self.n_parties = args.n_parties
        self.fraction = args.frac
        self.client_idx = np.arange(self.n_parties)
        self.args = args
        self.repeat_num = 0 if args.dataset in ('cifar10-dvs', 'dvs128gesture', 'nmnist', 'ncaltech101') else args.T
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.logger = Logger(args)
        self.best_acc = 0
        self.hete_T = args.hete_T

    def train(self):
        for ge in tqdm(range(self.global_epochs)):
            if self.args.layer_sample:
                sampled_users = stratified_sampling_indices(self.client_idx,
                                                            classes=np.arange(self.args.num_hete_models),
                                                            class_counts=np.multiply(self.args.portion, self.args.n_parties).astype(np.int32))
            else:
                np.random.shuffle(self.client_idx)
                sampled_users = self.client_idx[:int(self.n_parties * self.fraction)]
            # self.upload_parameters = [OrderedDict() for _ in range(len(sampled_users))]
            self.local_models, self.local_parameters, hete_ids = self.hete.distribute(self.global_model,self.client_idx,ge)
            # st = time.time()
            for net_id in sampled_users:
                local_model = self.local_models[net_id]
                local_model.train()
                local_model.load_state_dict(self.local_parameters[net_id])
                local_model.to(self.device)
                local_tr_loader = DataLoader(
                    self.dist_tr_dataset[net_id],
                    shuffle=True,
                    batch_size=self.args.batch_size
                )
                optimizer = optim.SGD(local_model.parameters(), lr=self.args.lr)
                for le in range(self.local_epochs):
                    for i,batch in enumerate(local_tr_loader):
                        x, target = batch
                        x, target = x.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        target = target.long()
                        if self.repeat_num:
                            x = x.unsqueeze(0).repeat(self.hete_T[hete_ids[net_id]], 1, 1, 1, 1)  # ->(T, B, C, H, W)
                        else:  # dvs data
                            x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                        out = local_model(x)
                        loss = self.criterion(out, target)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1)
                        optimizer.step()
                        functional.reset_net(local_model)
                local_model.to("cpu")
                self.local_parameters[net_id] = local_model.state_dict()
            # et = time.time()
            # print("train time consume:",et-st)
            self.aggregate(sampled_users)
            del self.local_parameters
            del self.local_models
            if ge % self.args.log_round == 0:
                # st = time.time()
                self.evaluate()
                # et = time.time()
                # print("evaluate time consume:",et-st)
        pass

    def evaluate(self):
        dist_ts_loader = torch.utils.data.DataLoader(dataset=self.dist_ts, batch_size=64, shuffle=False)
        self.global_model.to(self.device)
        test_acc = compute_accuracy(self.global_model,dist_ts_loader,device=self.device,repeat_num=self.repeat_num)
        if test_acc > self.best_acc:
            self.logger.info(f"net {self.global_model.model} Test accuracy: {test_acc:.4f}")
        self.global_model.to("cpu")
        pass

    def aggregate(self,sampled_users):
        self.hete.aggregate(self.global_model,self.local_parameters,sampled_users)
        pass

# knowledge optimized based fedavg
# this is absoutely great fed method
class kdfedavg_server(fedavg_server):
    def __init__(self,global_model,datasets,distributor,args):
        super(kdfedavg_server, self).__init__(global_model,datasets,distributor,args)
        self.kd_iterations = 10
        self.kd_lr = 1e-3
        self.softmax_temp = 1
        self.ce = torch.nn.CrossEntropyLoss()

    def aggregate(self,sampled_users):
        # local model contribute some to
        dist_ts_loader = torch.utils.data.DataLoader(dataset=self.dist_ts, batch_size=1, shuffle=True)
        self.hete.aggregate(self.global_model,self.local_parameters,self.client_idx)
        # del self.local_parameters
        self.global_model.to(self.device)
        self.global_model.train()
        optimizer = optim.SGD(self.global_model.parameters(),lr=self.kd_lr)
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
        for net_id in sampled_users:
            local_model = self.local_models[net_id]
            local_model.to(self.device)
            for i,batch in enumerate(dist_ts_loader):
                x, target = batch  # no label target
                x = x.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                if self.repeat_num:
                    x = x.unsqueeze(0).repeat(self.repeat_num, 1, 1, 1, 1)  # always (T, B, C, H, W)
                else:
                    x = x.transpose(0, 1)
                _output = self.global_model(x)
                loss = 0.01 * (self.softmax_temp ** 2) * criterion(
                    F.softmax(_output / self.softmax_temp,dim=1),
                    F.log_softmax(local_model(x).detach() / self.softmax_temp,dim=1)
                )
                # loss = self.ce(_output,target)
                loss.backward()
                optimizer.step()
                functional.reset_net(self.global_model)
                functional.reset_net(local_model)
                if i == self.kd_iterations:
                    break
            local_model.to("cpu")
        self.global_model.to("cpu")
        pass


# random shuffled avg
class rsfedavg_server(fedavg_server):
    def __init__(self,global_model,datasets,distributor,args):
        super(rsfedavg_server,self).__init__(global_model,datasets,distributor,args)
        self.args.strategy = "fed_rolex"  # change strategy, turn on something


# momentum based fedavg
# the parameter updated is being weightd by momentum
class mfedavg_server(fedavg_server):
    def __init__(self,global_model,datasets,distributor,args):
        super(mfedavg_server,self).__init__(global_model,datasets,distributor,args)
        self.dummy_models = []
        self.dummy_names = []
        self.leave_one_names = []
        self.args = args
        self.hete_ids = []
        self.optimizers = [optim.Adam(dummy_model.parameters(),lr=self.args.lr) for dummy_model in self.dummy_models]
        self.init_dummy_model()
        self.mom_beta = 0.2
        self.hete_exclude_name = []

    def init_dummy_model(self):
        dummy_names = []
        if "resnet" in self.args.model:
            for i,nb in enumerate(self.args.model_hete):
                model_name = f"resnet{nb*2+2}"
                self.args.model = model_name
                self.dummy_models.append(Aggregator_mom(i,self.args,self.args.num_train_sam))
        else:
            for i,nb in enumerate(self.args.model_hete):
                model_name = f"{self.args.model[:3]}{nb}"
                self.args.model = model_name
                self.dummy_models.append(Aggregator_mom(i,self.args,self.args.num_train_sam))
        for i,dummy_model in enumerate(self.dummy_models):
            self.dummy_names.append(
                [name for name, param in dummy_model.model.named_parameters() if param.requires_grad and "bias" not in name])
            # needs more concern here
            suffix = 3 if "vgg" in self.args.model else 1
            self.leave_one_names.append([name for name in dummy_names if get_block_id(name,i,self.args) == self.args.model_hete[i] - suffix])
        if "resnet" in self.args.model:
            self.hete_exclude_name.append("conv1.0.weight")
            self.hete_exclude_name.append("conv1.1.bn.weight")
            self.hete_exclude_name.append("conv1.1.bn.running_mean")
            self.hete_exclude_name.append("conv1.1.bn.running_var")
            self.hete_exclude_name.append("fc.weight")
        elif "vit" in self.args.model:
            self.hete_exclude_name.append("patch_embed.proj_conv.weight")
            self.hete_exclude_name.append("patch_embed.proj_conv2.weight")
            self.hete_exclude_name.append("patch_embed.rpe_conv.weight")
            self.hete_exclude_name.append("head.weight")
        elif "vgg" in self.args.model:
            self.hete_exclude_name.append("fc.1.weight")
            self.hete_exclude_name.append("fc.3.weight")
            self.hete_exclude_name.append("fc.5.weight")
        pass

    def aggregate(self,sampled_users):
        hete_models = [self.hete_ids[u] for u in sampled_users]
        idxs, unq_cnt = np.unique(hete_models, return_counts=True)
        did2_sample_portion = [0] * self.args.num_hete_models
        for i, idx in enumerate(idxs):
            did2_sample_portion[idx] = unq_cnt[i]
        tot = sum(did2_sample_portion)
        did2_sample_portion = [x / tot for x in did2_sample_portion]
        for i in range(self.args.num_hete_models):
            self.dummy_models[i].update()
        HeteAgg_mom(0, self.logger, self.args,
                    self.dummy_models, self.dummy_names,
                    self.leave_one_names,
                    self.hete_exclude_name,
                    did2_sample_portion)
        self.global_model.load_state_dict(self.dummy_models[-1].model.state_dict())
        pass

    def train(self):
        for ge in tqdm(range(self.global_epochs)):
            np.random.shuffle(self.client_idx)
            sampled_users = self.client_idx[:int(self.n_parties * self.fraction)]
            # self.upload_parameters = [OrderedDict() for _ in range(len(sampled_users))]
            self.local_models, self.local_parameters = self.hete.distribute(self.global_model,sampled_users,ge)
            # st = time.time()
            for net_id in sampled_users:
                local_model = self.local_models[net_id]
                local_model.train()
                local_model.load_state_dict(self.local_parameters[net_id])
                local_model.to(self.device)
                local_tr_loader = DataLoader(
                    self.dist_tr_dataset[net_id],
                    shuffle=True,
                    batch_size=self.args.batch_size
                )
                optimizer = optim.Adam(local_model.parameters(), lr=self.args.lr)
                optimizer.zero_grad()
                for le in range(self.local_epochs):
                    for i,batch in enumerate(local_tr_loader):
                        x, target = batch
                        x, target = x.to(self.device), target.to(self.device)
                        optimizer.zero_grad()
                        target = target.long()
                        if self.repeat_num:
                            x = x.unsqueeze(0).repeat(self.args.T, 1, 1, 1, 1)  # ->(T, B, C, H, W)
                        else:  # dvs data
                            x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                        out = local_model(x)
                        loss = self.criterion(out, target)
                        loss.backward()
                        model_grads = process_model_grad(local_model.named_parameters(),
                                                         self.args.batch_size / len(local_tr_loader))
                        self.dummy_models[self.hete_ids[net_id]].collect(model_grads)
                        dummy_state_dict = local_model.state_dict()
                        bn_status = {k: v for k, v in dummy_state_dict.items() if "running" in k}
                        if len(bn_status) == 0:
                            bn_status = None
                        self.dummy_models[self.hete_ids[net_id]].collect_bn_status(bn_status)
                        functional.reset_net(local_model)
            self.aggregate(sampled_users)
            del self.local_models
            del self.local_parameters
            if ge % self.args.log_round == 0:
                # st = time.time()
                self.evaluate()
                # et = time.time()
                # print("evaluate time consume:",et-st)
        pass


def get_block_id(name, hete_id, args, type="weight"):
    if 'layer' not in name and "vgg" not in args.model:
        return -1
    if "resnet" in args.model:
        pattern = re.compile(f'layer(\d+).(\d+).([a-z_A-Z]+).(\d+).{type}')
        full_name = pattern.findall(name)[0]
        layer_id = int(full_name[0]) - 1
        inter_blk_id = int(full_name[1])
        block_id = args.layer2_block[hete_id][layer_id] + inter_blk_id
    elif "vit" in args.model:  # vit
        pattern = re.compile(f'block.(\d+).')
        block_id = pattern.findall(name)[1]
        block_id = int(block_id)
    else:  # vgg
        pattern = re.compile(f'layer(\d+).(\d+).(\d+).weight')
        pattern_str = pattern.findall(name)[0]
        layer_id = pattern_str[0]
        inter_blk_id = pattern_str[1]
        block_id = args.layer2_block[hete_id][layer_id] + inter_blk_id
    return block_id