import argparse
from model_hete_utils import hete_scenarios
from utils import load_yaml,merge_args_yaml,CLASS_NUM,load_data,get_dataloader
import torch
from torch.utils.data import DataLoader,Dataset
import random
import numpy as np
from sampler import partition_data
from models.resnet import Resnet
from models.vgg import Staged_VGG
from models.vit import SViT
from feds import fed_aggs

def parse_args():
    parser = argparse.ArgumentParser()
    # common settings
    parser.add_argument('-data_dir', type=str, default='.', help='dataset path')
    parser.add_argument('-log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('-result_dir', type=str, required=False, default="./saved/", help='Model directory path')
    parser.add_argument('-snn', action='store_true', help="Whether to train SNN or ANN")
    parser.add_argument('-dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('-gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('-seed', '--init_seed', type=int, default=2025, help='Random seed')
    parser.add_argument("-has_rate", type=bool, default=False)
    parser.add_argument("-trs", type=bool, default=False)
    parser.add_argument("-rate", type=float, default=1.)
    parser.add_argument("-partition",type=str,default="iid")
    parser.add_argument("-hete_config", type=str, help="config file for model heterogeneity config")
    parser.add_argument("-alpha",type=float,default=0.1)
    parser.add_argument("-agg_config", type=str, default="fedavg",help="config file for agg algorithm config")
    parser.add_argument('-n_parties', type=int, default=100, help="number of users in total")
    parser.add_argument('-b','--batch_size',type=int, default=64, help="batch size")
    parser.add_argument('-frac', type=float, default=1.,
                        help='Sample ratio [0., 1.] of parties for each local training round')
    parser.add_argument("-local_epochs",type=int,default=3)
    parser.add_argument('-log_round', type=int, default=1, help='frequency of evaluating')
    parser.add_argument('-strategy',type=str,default="fedavg",help="0:fedavg,1:rsfedavg,2:mfedavg,3:kdfedavg")
    parser.add_argument("-fed_algo",type=int,default=0)
    parser.add_argument("-T",type=int,default=4,help="global model Time steps")
    parser.add_argument("-hete_T", nargs='+', type=int)
    parser.add_argument("-layer_sample", action='store_true')
    args = parser.parse_args()
    return args
    # extra model settings


def split_data(args):
    X_train, y_train, X_test, y_test = load_data(args)

    net_dataidx_map = partition_data(args, y_train)

    traindata_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        traindata_cls_counts[net_i] = tmp
    # logger.info(f'Finish partitioning, data statistics: {str(traindata_cls_counts)}')

    client_distribution = np.zeros((args.n_parties, args.num_classes))
    for i in range(args.n_parties):
        for j in traindata_cls_counts[i].keys():
            client_distribution[i][j] = traindata_cls_counts[i][j] / len(net_dataidx_map[i])

    train_all_in_list = []  # used to store local train dataset
    test_all_in_list = []  # used to store local test dataset
    noise_level = 0  # init noise level

    for party_id in range(args.n_parties):
        dataidxs = net_dataidx_map[party_id]
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
            (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, 0, data_name=args.dataset)

        train_all_in_list.append(train_ds_local)
        test_all_in_list.append(test_ds_local)

    test_ds_global = torch.utils.data.ConcatDataset(test_all_in_list)
    datasets = dict()
    datasets['train'] = train_all_in_list
    datasets['test'] = test_ds_global
    return datasets


def main():
    args = parse_args()
    basic_configs = load_yaml("./configs/train_config.yaml")
    hete_configs = load_yaml(args.hete_config)
    if args.agg_config != "fedavg":
        agg_configs = load_yaml(args.agg_config)
        merge_args_yaml(args, agg_configs)
    merge_args_yaml(args,basic_configs)
    merge_args_yaml(args, hete_configs)
    # split dataset into iid

    args.num_classes = CLASS_NUM[args.dataset]
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # reproducibility
    seed = args.init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    datasets = split_data(args)
    distributor = hete_scenarios[args.hete_id](args)
    if "resnet" in args.model:
        global_model = Resnet(args)
    elif "vgg" in args.model:
        global_model = Staged_VGG(args)
    else:
        global_model = SViT(args)
    # fed_sever = fedavg_server(global_model,datasets,distributor,args)
    # fed_sever = mfedavg_server(global_model, datasets, distributor, args)

    # def generate_probability_array_with_dirichlet(alpha, length):
    #     # 使用Dirichlet分布生成概率数组
    #     probability_array = np.random.dirichlet([alpha] * length)
    #
    #     # 确保每个值大于0.1
    #     min_prob = 0.1
    #     if np.any(probability_array < min_prob):
    #         # 剩余的概率
    #         remaining_prob = 1 - min_prob * length
    #         # 生成一个符合条件的随机数组
    #         random_array = np.random.rand(length)
    #         random_array = random_array / random_array.sum() * remaining_prob
    #         probability_array = random_array + min_prob
    #
    #     return probability_array
    #
    # portion = generate_probability_array_with_dirichlet(args.alpha, args.num_hete_models)
    # args.portion = portion
    fed_server = fed_aggs[args.fed_algo](global_model,datasets,distributor,args)
    fed_server.train()
    # layer hete, stage hete, hete hete
    # define levels
    # init models, global model and local model (based on scenarios), comes from config file
    # global models to evaluate: resnet34, vgg11, vit4
    # init fed methods, avg,kd-based avg,random shuffled avg, momentum-based avg
    # init training settings
    # split datasets, iid split would be enough
    # start training
    # evaluate
    pass


if __name__ == "__main__":
    main()
