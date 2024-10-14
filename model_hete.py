import random
import argparse

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from sampler import partition_data
from utils import *

from models.vit import SViT
from models.resnet import Resnet
# from models.raw_resnet import Resnet
from models.vgg import Staged_VGG
from model_hete_utils.in_fl_utils import Aggregator_mom,process_model_grad,HeteAgg_mom
import re
from model_hete_utils.Inco_agg import *
import copy

parser = argparse.ArgumentParser()
# common settings
parser.add_argument('-data_dir', type=str, default='.', help='dataset path')
parser.add_argument('-log_dir', type=str, required=False, default="./logs/", help='Log directory path')
parser.add_argument('-result_dir', type=str, required=False, default="./saved/", help='Model directory path')
parser.add_argument('-model', type=str, default='vgg', help='neural network used in training')
parser.add_argument('-snn', action='store_true', help="Whether to train SNN or ANN")
parser.add_argument('-dataset', type=str, default='cifar10', help='dataset used for training')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('-T', type=int, default=4, help='time step for SNN neuron (default: 4)')
parser.add_argument('-lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
parser.add_argument('-optimizer', type=str, default='adam', help='the optimizer')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help='weight decay for optimizer')
parser.add_argument('-momentum', type=float, default=0.9, help='Parameter controlling the momentum SGD')
parser.add_argument('-gpu', type=int, default=0, help="GPU ID, -1 for CPU")
parser.add_argument('-seed', '--init_seed', type=int, default=2025, help='Random seed')
# federate settings
parser.add_argument("-has_rate",type=bool,default=False)
parser.add_argument("-trs",type=bool,default=False)
parser.add_argument("-rate",type=float,default=1.)
# parser.add_argument('-model_split_mode',type=str,default="fix",help="model split mode for hetero_fl")
# parser.add_argument("-model_mode",type=str,default="None",help="model mode for hetero_fl")
parser.add_argument("-config",type=str,help="config file for model heterogeneity config")
parser.add_argument('-noise', type=float, default=0, help='how much noise we add to some party')  # 0.1
parser.add_argument('-noise_type', type=str, default='level',
                    help='Different level or space of noise (Optional: level, space)')
parser.add_argument('-partition', type=str, default='iid', help='the data partitioning strategy')
parser.add_argument('-alpha', type=float, default=0.5,
                    help='The parameter for the dirichlet distribution for data partitioning')
parser.add_argument('-strategy', type=str, default='fedavg', help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
parser.add_argument('-n_parties',type=int,default=100,help="number of users in total")
parser.add_argument('-frac', type=float, default=1.,
                    help='Sample ratio [0., 1.] of parties for each local training round')
parser.add_argument('-global_epochs', type=int, default=1000, help='rounds of updating global model')
parser.add_argument('-InCo_training', action="store_true", help='whether to use Inco_training')
parser.add_argument('-mu', type=float, default=0.01, help='the regularization parameter for fedprox')
parser.add_argument('-local_epochs', type=int, default=10, help='number of local training rounds')
parser.add_argument('-lc', action='store_true', help="Whether to do loss calibration tackling label skew")
parser.add_argument('-rs', action='store_true', help="Whether to do restricted softmax tackling label skew")
parser.add_argument('-tune_epochs', type=int, default=1000, help='Classifier communication round for FedConcat')
parser.add_argument('-eps', type=float, default=0,
                    help='Epsilon for differential privacy to protect label distribution')
parser.add_argument('-tau', type=float, default=0.1, help='calibration loss constant for fedLC')
parser.add_argument('-log_round', type=int, default=1, help='frequency of evaluating')
parser.add_argument('-not_same_initial', action='store_true',
                    help='Whether initial all the models with the same parameters in fedavg')

# extra model settings
parser.add_argument('-nh', '--num_heads', type=int, default=8, help='number of attention heads for spikeformer')
parser.add_argument('-dim', '--hidden_dim', type=int, default=384,
                    help='dimension of intermediate layer in spikeformer FFN part')
parser.add_argument("-hete_T",nargs='+',type=int)
parser.add_argument("-layer_sample",action="store_true")

args = parser.parse_args()

args.num_classes = CLASS_NUM[args.dataset]
args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
if args.device != 'cpu':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device(args.device)
repeat_num = 0 if args.dataset in ('cifar10-dvs', 'dvs128gesture', 'nmnist', 'ncaltech101') else args.T

# reproducibility
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

logger = Logger(args)
logger.info(str(args))

logger.info("Loading data")
X_train, y_train, X_test, y_test = load_data(args)

# partitioning data to different cluster
logger.info("Partitioning data")
net_dataidx_map = partition_data(args,y_train)

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

# differential privacy
if args.eps > 0:
    for i in range(args.n_parties):
        lap = np.random.laplace(0, 1 / args.eps, args.num_classes)
        for j in range(args.num_classes):
            client_distribution[i][j] += lap[j]

    logger.info(client_distribution)

train_all_in_list = []  # used to store local train dataset
test_all_in_list = []  # used to store local test dataset
noise_level = 0  # init noise level

for party_id in range(args.n_parties):
    dataidxs = net_dataidx_map[party_id]
    noise_level = args.noise

    if party_id == args.n_parties - 1:
        noise_level = 0  # reset

    if args.noise == 0:  # no noise
        train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
            (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, 0, data_name=args.dataset)
    else:
        # in any case when noise=0 they all will return all local data are sampled without noise
        if args.noise_type == 'space':  # add noise to certain pixel points, image space noise
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, noise_level, party_id,
                args.n_parties - 1, data_name=args.dataset)
        else:  # noise-based feature imbalance, level
            noise_level = args.noise / (args.n_parties - 1) * party_id
            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(
                (X_train, X_test), (y_train, y_test), args.batch_size, 32, dataidxs, noise_level,
                data_name=args.dataset)

    train_all_in_list.append(train_ds_local)
    test_all_in_list.append(test_ds_local)

train_ds_global = torch.utils.data.ConcatDataset(train_all_in_list)
test_ds_global = torch.utils.data.ConcatDataset(test_all_in_list)
test_dl_global = torch.utils.data.DataLoader(dataset=test_ds_global, batch_size=64, shuffle=False)
logger.info(f'length of train_dl_global: {len(train_ds_global)}')

logger.info('Init models')
criterion = nn.CrossEntropyLoss().to(device)

arr = np.arange(args.n_parties)



if args.strategy == "inclusive_fl":
    # default inclusive fl args
    configs = load_yaml(args.config)
    configs['model_hete'] = configs['model_hete'][0]
    # configs['hete_ids'] = configs['hete_ids'][0]
    merge_args_yaml(args,configs)
    portion = args.portion
    # portion = generate_probability_array_with_dirichlet(args.alpha, args.num_hete_models)
    # args.has_rate = False
    # args.portion = (np.ones(args.num_hete_total) * (1.0 / args.num_hete_total)).tolist()
    # args.drop_idx = []
    # args.local_one = True
    # args.mom_beta = 0.2
    # resnet_hete = [4,8,16]  # resnet10, resnet18, resnet34, based on hidden blocks
    # vit_hete = [1,2,4]        #
    model_names = []
    hete_exclude_name = []
    users2_model = []
    best_acc = 0
    # just aggregate bn.running_var and bn.running_mean by mean
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
            layer_id = int(pattern_str[0])
            inter_blk_id = int(pattern_str[1])
            block_id = args.layer2_block[hete_id][layer_id] + inter_blk_id
        return block_id
    # def get_block_id(name, hete_id, type="weight"):
    #     if 'layer' not in name and "vgg" not in args.model:
    #         return -1
    #     if "resnet" in args.model:
    #         # print(name)
    #         # if "bn" in name:
    #         #     pattern = re.compile(f'layer(\d+).(\d+).([A-Za-z0-9]+).(\d+).bn.{type}')
    #         # else:
    #         #     pattern = re.compile(f'layer(\d+).(\d+).([A-Za-z0-9]+).(\d+).{type}')
    #         pattern = re.compile(f'layer(\d+).(\d+).([a-z_A-Z]+).(\d+).{type}')
    #         full_name = pattern.findall(name)[0]
    #         layer_id = int(full_name[0]) - 1
    #         inter_blk_id = int(full_name[1])
    #         block_id = args.layer2_block[hete_id][layer_id] + inter_blk_id
    #     elif "vit" in args.model:  # vit
    #         pattern = re.compile(f'block.(\d+).')
    #         block_id = pattern.findall(name)[1]
    #         block_id = int(block_id)
    #     else:  # vgg
    #         vgg_cfg = {
    #             'VGG5': [64, 'P', 128, 128, 'P'],
    #             'VGG9': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'],
    #             'VGG11': [64, 'P', 128, 'p', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    #             'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    #             'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    #             'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512,
    #                       512, 'P']
    #         }
    #         pattern = re.compile(f'([a-z_A-Z]+).(\d+).weight')
    #         block_id = pattern.findall(name)[0][1]
    #         block_id = int(block_id)
    #         layer_num = args.model_hete[hete_id]
    #         cfg = vgg_cfg[f'VGG{layer_num}']
    #         layer_count = 0
    #         real_layer = 0
    #         for x in cfg:
    #             if x == "P":
    #                 layer_count += 1
    #             else:
    #                 layer_count += 3
    #                 real_layer += 1
    #             if block_id == layer_count:
    #                 break
    #         block_id = real_layer
    #     return block_id

    def init_nets(n_parties, args):
        dummy_models = []
        # args.portion  = []
        # args.num_hete_models
        # hete_nums = args.portion
        # print(hete_nums)
        hete_ids = []
        for i,num in enumerate(portion):
            num = math.ceil(args.n_parties * num)
            for _ in range(num):
                hete_ids.append(i)
        # hete_ids = args.hete_ids   # [0,0,0,0,1,1,1,1,2,2]
        for net_i in range(n_parties):
            # randomly choose a hete model from 3
            hete_id = hete_ids[net_i]
            users2_model.append(hete_id)
            if 'resnet' in args.model:
                # hidden_blocks = resnet_hete
                # args.num_hidden_layers = resnet_hete
                hidden_blocks = args.model_hete  # [4,6,8]
                args.num_hidden_layers = args.model_hete
                model_name = "resnet" + str(hidden_blocks[hete_id] * 2 + 2)
                args.model = model_name
                net = Resnet(args)
            elif 'vit' in args.model:
                # hidden_blocks = vit_hete
                # args.num_hidden_layers = vit_hete
                hidden_blocks = args.model_hete
                args.num_hidden_layers = args.model_hete
                model_name = "vit" + str(hidden_blocks[hete_id])
                args.model = model_name
                net = SViT(args)
            else:
                hidden_blocks = args.model_hete
                args.num_hidden_layers = args.model_hete
                model_name = "vgg" + str(hidden_blocks[hete_id])
                args.model = model_name
                net = Staged_VGG(args)
            net.to(device)
            dummy_models.append(net)
            del net
        return dummy_models


    dummy_models = init_nets(args.n_parties,args)
    dummy_names = []
    leave_one_names = []
    global_acc_record = []
    if "resnet" in args.model:
        hete_exclude_name.append("conv1.0.weight")
        hete_exclude_name.append("conv1.1.bn.weight")
        hete_exclude_name.append("conv1.1.bn.running_mean")
        hete_exclude_name.append("conv1.1.bn.running_var")
        hete_exclude_name.append("fc.weight")
    elif "vit" in args.model:
        hete_exclude_name.append("patch_embed.proj_conv.weight")
        hete_exclude_name.append("patch_embed.proj_conv2.weight")
        hete_exclude_name.append("patch_embed.rpe_conv.weight")
        hete_exclude_name.append("head.weight")
    elif "vgg" in args.model:
        hete_exclude_name.append("fc.1.weight")
        hete_exclude_name.append("fc.3.weight")
        hete_exclude_name.append("fc.5.weight")
    num_train_sam = len(train_ds_global)
    aggs = []
    for i in range(args.num_hete_models):
        if "resnet" in args.model:
            # args.model = "resnet" + str(resnet_hete[i] * 2 + 2)
            args.model = "resnet" + str(args.model_hete[i] * 2 + 2)
            aggs.append(Aggregator_mom(i, args, num_train_sam))
        elif "vit" in args.model:
            # args.model = "vit" + str(vit_hete[i])
            args.model = "vit" + str(args.model_hete[i])
            aggs.append(Aggregator_mom(i, args, num_train_sam))
        elif "vgg" in args.model:
            args.model = "vgg" + str(args.model_hete[i])
            aggs.append(Aggregator_mom(i, args, num_train_sam))
        # dummy_names.append(
        #     [k for k in aggs[i].model.state_dict().keys() if
        #      "bias" not in k and "num_batches_tracked" not in k])
        res = 4 if "vgg" in args.model else 1
        dummy_names.append(
            [name for name, param in aggs[i].model.named_parameters() if param.requires_grad and "bias" not in name])
        leave_one_names.append(
            [name for name in dummy_names[i] if "fc" not in name and get_block_id(name, i, args,type=name.split(".")[-1]) == args.num_hidden_layers[i] - res])
        # num of aggregators
    Hete_T = args.hete_T
    for r in tqdm(range(args.global_epochs)):
        # randomly select users for this round

        if args.layer_sample:
            sampled_users = stratified_sampling_indices(arr,
                                                        classes=np.arange(args.num_hete_models),
                                                        class_counts=np.multiply(portion,args.n_parties).astype(np.int32))
        else:
            np.random.shuffle(arr)
            sampled_users = arr[:int(args.n_parties * args.frac)]
        hete_models = [users2_model[u] for u in sampled_users]
        idxs,unq_cnt = np.unique(hete_models,return_counts=True)
        did2_sample_portion = [0] * args.num_hete_models
        # for i in range(args.num_hete_models):
        #     did2_sample_portion.append(unq_cnt[i])
        for i,idx in enumerate(idxs):
            did2_sample_portion[idx] = unq_cnt[i]
        tot = sum(did2_sample_portion)
        did2_sample_portion = [x / tot for x in did2_sample_portion]
        for net_id in sampled_users:
            # print("******* fed train over {}-{:.2f} portion of users ******".format(i, did2_sample_portion[i]))
            train_dataset_i = train_all_in_list[net_id]
            train_dataloader = DataLoader(
                train_dataset_i,
                shuffle=True,
                batch_size=args.batch_size
            )
            avg_loss = 0
            cnt = 0
            dummy_models[net_id].load_state_dict(aggs[users2_model[net_id]].model.state_dict())
            dummy_models[net_id].train()
            dummy_optimizer = optim.Adam(dummy_models[net_id].parameters(), lr=args.lr)
            for _ in range(args.local_epochs):
                for step, batch in enumerate(train_dataloader):
                    x, target = batch
                    x, target = x.to(device), target.to(device)
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(Hete_T[users2_model[net_id]], 1, 1, 1, 1)  # -> (T, B, C, H, W)
                    else:  # dvs data
                        x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                    target = target.long()
                    x.requires_grad = True
                    target.requires_grad = False

                    outputs = dummy_models[net_id](x)
                    loss = criterion(outputs, target)

                    dummy_optimizer.zero_grad()
                    loss.backward()
                    model_grads = process_model_grad(dummy_models[net_id].named_parameters(),
                                                     args.portion[users2_model[net_id]] * args.n_parties)
                    torch.nn.utils.clip_grad_norm_(dummy_models[net_id].parameters(), 1)
                    aggs[users2_model[net_id]].collect(model_grads)
                    dummy_optimizer.step()
                    functional.reset_net(dummy_models[net_id])
                # aggs[users2_model[net_id]].update()
                # if r > 100:
                #     test_acc = compute_accuracy(dummy_models[net_id], test_dl_global, device=device,
                #                                 repeat_num=repeat_num)
                #     print(test_acc)
            dummy_state_dict = dummy_models[net_id].state_dict()
            bn_status = {k: v for k, v in dummy_state_dict.items() if "running" in k}
            if len(bn_status) == 0:
                bn_status = None
            aggs[users2_model[net_id]].collect_bn_status(bn_status)
        for i in range(args.num_hete_models):
            aggs[i].update()

        if r % args.log_round == 0:
            test_acc = compute_accuracy(aggs[-1].model, test_dl_global, device=device, repeat_num=repeat_num)
            if test_acc > best_acc:
                best_acc = test_acc
            logger.info(f"net {args.model} Test accuracy: {test_acc:.4f}")
            pass

            # in loop end
            # print("******* end fed train {} with {} users ******".format(args.task_name, did2_sample_portion))

        # aggregation
        HeteAgg_mom(r, logger, args, aggs, dummy_names, leave_one_names, hete_exclude_name, did2_sample_portion)
    logger.info(f"net {args.model} Best Test Accuracy:{best_acc:.4f}")
elif args.strategy == "hetero_fl":
    configs = load_yaml(args.config)
    merge_args_yaml(args,configs)
    # portion = generate_probability_array_with_dirichlet(args.alpha, args.num_hete_models)
    best_acc = 0
    all_label_split = {}
    for u in range(args.n_parties):
        all_label_split[u] = np.unique(y_train[net_dataidx_map[u]]).tolist()
        all_label_split[u] = torch.tensor(all_label_split[u])
    from model_hete_utils.hetero_fl_utils import *
    if "resnet" in args.model:
        model = Resnet
    elif "vit" in args.model:
        model = SViT
    else:
        model = Staged_VGG
    global_model = model(args).to(device)
    client_models = {}
    model_rate = args.mode_rate
    portion = (np.array(args.portion) / sum(args.portion)).tolist()
    classes_count = np.multiply(args.n_parties, list(reversed(portion))).astype(np.int32)
    rate_idx = []
    for idx,num in enumerate(classes_count):
        for _ in range(num):
            rate_idx.append(idx)
    # rate_idx = torch.multinomial(torch.tensor(portion), num_samples=args.n_parties,
    #                              replacement=True).tolist()
    model_rate = np.array(model_rate)[rate_idx]
    for i in range(args.n_parties):
        args.has_rate = True
        args.rate = model_rate[i]
        client_model = model(args).to(device)  # todo list about scale
        client_model.train()
        client_models[i] = client_model
    tmp_counts = {}
    for k, v in global_model.state_dict().items():
        tmp_counts[k] = torch.ones_like(v)
    for global_epoch in tqdm(range(args.global_epochs)):
        global_parameters = global_model.state_dict()
        # np.random.shuffle(arr)
        # sampled_users = arr[:int(args.n_parties * args.frac)]
        if args.layer_sample:
            sampled_users = stratified_sampling_indices(arr,
                                                        classes=np.arange(args.num_hete_models),
                                                        class_counts=np.multiply(portion,args.n_parties).astype(np.int32))
        else:
            np.random.shuffle(arr)
            sampled_users = arr[:int(args.n_parties * args.frac)]
        # begin split local models based on model rate
        # print(f"current model rate:{model_rate}")
        param_idx = split_model(global_parameters,model_rate,arr,global_epoch,args)
        # no idea about this
        local_parameters = [OrderedDict() for _ in range(len(arr))]
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
        # param_ids = [local_parameter for local_parameter in local_parameters]
        for u_id in sampled_users:
            client_model_rate = model_rate[u_id]
            train_dataset_i = train_all_in_list[u_id]
            train_dataloader_i = DataLoader(
                train_dataset_i,
                shuffle=True,
                batch_size=args.batch_size
            )
            client_parameters = local_parameters[u_id]
            client_model = client_models[u_id]
            client_model.load_state_dict(client_parameters)
            optimizer = optim.Adam(client_model.parameters(),lr=args.lr)
            label_split = all_label_split[u_id]
            for local_epoch in range(args.local_epochs):
                # step
                for i,batch in enumerate(train_dataloader_i):
                    x,target = batch
                    x, target = x.to(device), target.to(device)
                    optimizer.zero_grad()
                    target = target.long()
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.hete_T[rate_idx[u_id]], 1, 1, 1, 1)  # ->(T, B, C, H, W)
                    else:  # dvs data
                        x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                    out = client_model(x)
                    label_mask = torch.zeros(args.num_classes, device=out.device)
                    label_mask[label_split] = 1
                    out = out.masked_fill(label_mask == 0, 0)
                    loss = criterion(out,target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
                    optimizer.step()
                    functional.reset_net(client_model)
                    pass
            # test_acc = compute_accuracy(client_model, test_dl_global, device, repeat_num)
            local_parameters[u_id] = client_model.state_dict()
        # aggregation
        # global step
        count = OrderedDict()
        tmp_counts_cpy = copy.deepcopy(tmp_counts)
        updated_parameters = copy.deepcopy(global_model.state_dict())
        if "resnet" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(sampled_users)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'fc' in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'fc' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                            else:
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
        elif "vit" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(sampled_users)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if "patch_embed" in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                elif 'head' in k:
                                    label_split = all_label_split[sampled_users[sampled_users[m]]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'head' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                            else:
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
        elif "vgg" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(sampled_users)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'fc' in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'fc' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                            else:
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
        if global_epoch % args.log_round == 0:
            test_acc = compute_accuracy(global_model, test_dl_global, device=device, repeat_num=repeat_num)
            if test_acc > best_acc:
                best_acc = test_acc
            logger.info(f"net {args.model} acc:{test_acc:.4f}")
    logger.info(f"net {args.model} best acc:{best_acc:.4f}")
    pass

elif args.strategy == "scale_fl":  # remains some discussion
    # early exit mechanism
    from model_hete_utils.scale_fl_utils import get_downscale_index,get_local_split,fix_idx_array,KDLoss
    from models.ee_resnet import resnet34_4
    from models.ee_vit import SViT
    configs = load_yaml(args.config)
    merge_args_yaml(args,configs)
    if "resnet" in args.model:
        model = resnet34_4
    elif "vit" in args.model:
        model = SViT
    else:
        model = VGG
    global_model = model(args).to(device)
    idx_dicts = [get_downscale_index(global_model,args,s) for s in args.vertical_scale_ratios]
    # prepare client groups
    client_idxs = np.arange(args.n_parties)
    np.random.seed(args.init_seed)
    shuffled_client_idxs = np.random.permutation(client_idxs)
    client_groups = []
    s = 0

    def get_level(client_idx):
        try:
            level = np.where([client_idx in c for c in client_groups])[0][0]
        except:
            # client will be skipped
            level = -1
        return level

    for ratio in args.client_split_ratios:
        e = s + int(len(shuffled_client_idxs) * ratio)
        client_groups.append(shuffled_client_idxs[s: e])
        s = e
    client_groups = client_groups
    criterion = KDLoss(args).cuda(device)
    model_kwargs = global_model.stored_inp_kwargs
    local_models = [type(global_model)(**model_kwargs) for i in range(args.n_parties)]
    for global_epoch in tqdm(range(args.global_epochs)):
        np.random.shuffle(arr)
        sampled_users = arr[:int(args.n_parties * args.frac)]
        levels = [get_level(u_id) for u_id in sampled_users]
        scales = [args.vertical_scale_ratios[level] for level in levels]
        local_models = [get_local_split(global_model,idx_dicts,levels[i], scales[i]) for i in range(len(sampled_users))]
        h_scale_ratios = [args.horizontal_scale_ratios[level] for level in levels]
        all_local_weights = []
        # all_local_losses = []
        all_local_grad_flags = []

        for i,client_idx in enumerate(client_idxs):
            # begin local round
            local_model = local_models[i].to(device)
            local_model.train()
            base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
            exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]
            optimizer = torch.optim.SGD([{'params': base_params},
                                         {'params': exit_params}],
                                        lr=args.lr,
                                        momentum=args.momentum)
            train_dataset_i = train_all_in_list[client_idx]
            train_dataloader_i = DataLoader(
                train_dataset_i,
                shuffle=True,
                batch_size=args.batch_size
            )
            for local_epoch in range(args.local_epochs):
                for i,batch in enumerate(train_dataloader_i):
                    x, target = batch
                    x,target = x.to(device),target.to(device)
                    target = target.long()
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # ->(T, B, C, H, W)
                    else:  # dvs data
                        x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                    output = local_model(x,manual_early_exit_index=h_scale_ratios[i])
                    # output = local_model(x)
                    if not isinstance(output, list):
                        output = [output]
                    loss = 0.0
                    for j in range(len(output)):
                        if j == len(output) - 1:
                            loss += criterion.ce_loss(output[j], target) * (j + 1)
                        else:
                            gamma_active = global_epoch > args.global_epochs * 0.25
                            loss += criterion.loss_fn_kd(output[j], target, output[-1], gamma_active) * (j + 1)
                    optimizer.zero_grad()
                    loss /= len(output) * (len(output) + 1) / 2
                    loss.backward()
                    optimizer.step()
                    functional.reset_net(local_model)
                    # no need for log now
                    del output
            local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
            # local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}
            local_model.to(device)
            torch.cuda.empty_cache()
            # for k, v in local_weights.items():
            #     local_weights[k] = v.to(device)
            all_local_weights.append(local_weights)
            # all_local_grad_flags.append(local_grad_flags)
        # aggregation starts
        global_weights = copy.deepcopy(global_model.state_dict())
        for key in global_weights.keys():

            if 'num_batches_tracked' in key:
                global_weights[key] = all_local_weights[0][key]
                continue

            # if 'running' in key:
            #     # print(global_weights[key].shape)
            #     # for w_ in all_local_weights:
            #     #     print(w_[key].shape)
            #     global_weights[key] = sum([w_[key] for w_ in all_local_weights]) / len(all_local_weights)
            #     continue

            tmp = torch.zeros_like(global_weights[key])
            count = torch.zeros_like(tmp)
            for i in range(len(all_local_weights)):
                # if all_local_grad_flags[i][key]:
                #     idx = idx_dicts[levels[i]][key]
                #     idx = fix_idx_array(idx, all_local_weights[i][key].shape)
                #     tmp[idx] += all_local_weights[i][key].flatten()
                #     count[idx] += 1
                # else:
                idx = idx_dicts[levels[i]][key]
                idx = fix_idx_array(idx, all_local_weights[i][key].shape)
                tmp[idx] += all_local_weights[i][key].to(device).flatten()
                count[idx] += 1
            global_weights[key][count != 0] = tmp[count != 0]
            count[count == 0] = 1
            global_weights[key] = global_weights[key] / count
        global_model.load_state_dict(global_weights)
        if global_epoch % args.log_round == 0:
            test_acc = compute_accuracy(global_model, test_dl_global, args.strategy, device, repeat_num)
            logger.info(f"net server Test accuracy: {test_acc:.4f}")

elif args.strategy == "fed_rolex":
    configs = load_yaml(args.config)
    merge_args_yaml(args,configs)
    all_label_split = {}
    for u in range(args.n_parties):
        all_label_split[u] = np.unique(y_train[net_dataidx_map[u]]).tolist()
        all_label_split[u] = torch.tensor(all_label_split[u])
    from model_hete_utils.hetero_fl_utils import *
    if "resnet" in args.model:
        model = Resnet
    elif "vit" in args.model:
        model = SViT
    else:
        model = VGG
    global_model = model(args).to(device)
    optim_global = optim.Adam(global_model.parameters(),lr=args.lr)
    # if "resnet" in args.model:
    #     mode_rate, proportion = [1, 0.5, 0.25, 0.125, 0.0625], [1, 1, 1, 1, 1]
    # elif "vit" in args.model:
    #     mode_rate, proportion = [1, 0.5,0.25,0.125,0.0625], [1,1,1,1,1]
    # [6, 10, 11, 18, 55]
    num_users_proportion = args.n_parties // sum(args.proportion)
    model_rate = args.mode_rate
    portion = (np.array(portion) / sum(portion)).tolist()
    rate_idx = torch.multinomial(torch.tensor(portion), num_samples=args.n_parties,
                                 replacement=True).tolist()
    model_rate = np.array(model_rate)[rate_idx]
    client_models = {}
    for i in range(args.n_parties):
        args.has_rate = True
        args.rate = model_rate[i]
        client_model = model(args).to(device)  # todo list about scale
        client_model.train()
        client_models[i] = client_model
    tmp_counts = {}
    for k, v in global_model.state_dict().items():
        tmp_counts[k] = torch.ones_like(v)
    best_acc = 0
    for global_epoch in tqdm(range(args.global_epochs)):
        global_parameters = global_model.state_dict()
        np.random.shuffle(arr)
        sampled_users = arr[:int(args.n_parties * args.frac)]
        # begin split local models based on model rate
        param_idx = split_model(global_parameters,model_rate,sampled_users,global_epoch,args)
        # no idea about this
        local_parameters = [OrderedDict() for _ in range(len(sampled_users))]
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
        # param_ids = [local_parameter for local_parameter in local_parameters]
        for u_id in sampled_users:
            client_model_rate = model_rate[u_id]
            train_dataset_i = train_all_in_list[u_id]
            train_dataloader_i = DataLoader(
                train_dataset_i,
                shuffle=True,
                batch_size=args.batch_size
            )
            client_parameters = local_parameters[u_id]
            client_model = client_models[u_id]
            client_model.load_state_dict(client_parameters)
            optimizer = optim.Adam(client_model.parameters(),lr=args.lr)
            label_split = all_label_split[u_id]
            for local_epoch in range(args.local_epochs):
                # step
                for i,batch in enumerate(train_dataloader_i):
                    x,target = batch
                    x,target = x.to(device),target.to(device)
                    optimizer.zero_grad()
                    target = target.long()
                    if repeat_num:
                        x = x.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)  # ->(T, B, C, H, W)
                    else:  # dvs data
                        x = x.transpose(0, 1)  # (B, T, C, H, W) -> (T, B, C, H, W)
                    out = client_model(x)
                    label_mask = torch.zeros(args.num_classes, device=out.device)
                    label_mask[label_split] = 1
                    out = out.masked_fill(label_mask == 0, 0)
                    loss = criterion(out,target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1)
                    optimizer.step()
                    functional.reset_net(client_model)
                    pass
            # acc = compute_accuracy(client_model,test_dl_global)
            # collect results
            local_parameters[u_id] = client_model.state_dict()
        # aggregation
        # global step
        count = OrderedDict()
        tmp_counts_cpy = copy.deepcopy(tmp_counts)
        updated_parameters = copy.deepcopy(global_model.state_dict())
        if "resnet" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'fc' in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += tmp_counts[k][torch.meshgrid(
                                        param_idx[sampled_users[m]][k])] * local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += tmp_counts[k][torch.meshgrid(
                                        param_idx[sampled_users[m]][k])]
                                    tmp_counts_cpy[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    output_size = v.size(0)
                                    scaler_rate = model_rate[sampled_users[m]]
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    # temp select avg
                                    # if self.cfg['weighting'] == 'avg':
                                    K = 1
                                    # elif self.cfg['weighting'] == 'width':
                                    #     K = local_output_size
                                    # elif self.cfg['weighting'] == 'updates':
                                    #     K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # elif self.cfg['weighting'] == 'updates_width':
                                    #     K = local_output_size * tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # K = local_output_size
                                    # K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += K * local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += K
                                    tmp_counts_cpy[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'fc' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k][
                                    label_split]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * \
                                                                 local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                        tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += tmp_counts[k] * local_parameters[sampled_users[m]][k]
                        count[k] += tmp_counts[k]
                        tmp_counts_cpy[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
                tmp_counts = tmp_counts_cpy
        elif "vit" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if 'weight' in parameter_type:
                            if v.dim() > 1:
                                if "patch_embed" in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                elif 'head' in k:
                                    label_split = all_label_split[sampled_users[sampled_users[m]]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'head' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k][label_split]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * \
                                                                 local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                        tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += local_parameters[sampled_users[m]][k]
                        count[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        elif "vgg" in args.model:
            for k, v in updated_parameters.items():
                parameter_type = k.split('.')[-1]
                count[k] = v.new_zeros(v.size(), dtype=torch.float32)
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    if 'weight' in parameter_type or 'bias' in parameter_type:
                        if parameter_type == 'weight':
                            if v.dim() > 1:
                                if 'fc' in k:
                                    label_split = all_label_split[sampled_users[m]]
                                    param_idx[sampled_users[m]][k] = list(param_idx[sampled_users[m]][k])
                                    param_idx[sampled_users[m]][k][0] = param_idx[sampled_users[m]][k][0][label_split]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += tmp_counts[k][torch.meshgrid(
                                        param_idx[sampled_users[m]][k])] * local_parameters[sampled_users[m]][k][label_split]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += tmp_counts[k][torch.meshgrid(
                                        param_idx[sampled_users[m]][k])]
                                    tmp_counts_cpy[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                                else:
                                    output_size = v.size(0)
                                    scaler_rate = model_rate[sampled_users[m]]
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                    # temp select avg
                                    # if self.cfg['weighting'] == 'avg':
                                    K = 1
                                    # elif self.cfg['weighting'] == 'width':
                                    #     K = local_output_size
                                    # elif self.cfg['weighting'] == 'updates':
                                    #     K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # elif self.cfg['weighting'] == 'updates_width':
                                    #     K = local_output_size * tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # K = self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    # K = local_output_size
                                    # K = local_output_size * self.tmp_counts[k][torch.meshgrid(param_idx[m][k])]
                                    tmp_v[torch.meshgrid(param_idx[sampled_users[m]][k])] += K * local_parameters[sampled_users[m]][k]
                                    count[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += K
                                    tmp_counts_cpy[k][torch.meshgrid(param_idx[sampled_users[m]][k])] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                        else:
                            if 'fc' in k:
                                label_split = all_label_split[sampled_users[m]]
                                param_idx[sampled_users[m]][k] = param_idx[sampled_users[m]][k][label_split]
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k][
                                    label_split]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                            else:
                                tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * local_parameters[sampled_users[m]][k]
                                count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                                tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                    elif "running" in parameter_type:
                        tmp_v[param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]] * \
                                                                 local_parameters[sampled_users[m]][k]
                        count[k][param_idx[sampled_users[m]][k]] += tmp_counts[k][param_idx[sampled_users[m]][k]]
                        tmp_counts_cpy[k][param_idx[sampled_users[m]][k]] += 1
                    else:
                        tmp_v += tmp_counts[k] * local_parameters[sampled_users[m]][k]
                        count[k] += tmp_counts[k]
                        tmp_counts_cpy[k] += 1
                tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
                v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
                tmp_counts = tmp_counts_cpy
        global_model.load_state_dict(updated_parameters)
        if global_epoch % args.log_round == 0:
            test_acc = compute_accuracy(global_model, test_dl_global, device=device, repeat_num=repeat_num)
            if test_acc > best_acc:
                best_acc = test_acc
            logger.info(f"net server Test accuracy: {test_acc:.4f}")
        # emerge
    logger.info(f"net {args.model} Best Test accuracy: {best_acc:.4f}")
    pass
