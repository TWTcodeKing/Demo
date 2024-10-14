import re

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate

from utils import INPUT_SIZE

vgg_cfg = {
    'VGG5' : [64, 'P', 128, 128, 'P'],
    'VGG9':  [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P'],
    'VGG11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512 , 'P', 512, 512, 'P'],
    'VGG16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
    'VGG19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']
}


class SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SimpleCNN, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        self.conv_fc = nn.Sequential(
            layer.Conv2d(C, 6, kernel_size=5),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(6, 16, kernel_size=5),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Linear((((H - 5 + 1) // 2 - 5 + 1) // 2) * (((W - 5 + 1) // 2 - 5 + 1) // 2) * 16, 120),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.Linear(120, 84),
            neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
            layer.Linear(84, args.num_classes),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.conv_fc(x)

        return x.mean(0)

class VGG(nn.Module):
    def __init__(self, args):
        ''' VGG '''
        super(VGG, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        self.rate = args.rate
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        conv = []
        in_channel = C

        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                conv.append(layer.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
            else:
                out_channel = int(x * self.rate)
                conv.append(
                    layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                conv.append(layer.BatchNorm2d(out_channel,track_running_stats=args.trs))
                conv.append(
                    neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU())
                in_channel = out_channel

        self.features = nn.Sequential(*conv)

        if int(layer_num) in [5, 9]:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, int(1024*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(int(1024*args.rate), args.num_classes, bias=self.bias_flag),
            )
        else:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, int(4096*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(int(4096*args.rate), int(4096*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(int(4096*args.rate), args.num_classes, bias=self.bias_flag),
            )
        # print(f"rate:{self.rate},channel:{in_channel},H:{H},W:{W}")

        # for m in self.modules():
        #     if isinstance(m, (layer.Conv2d, layer.Linear)):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=2)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.features(x)  # (T, B, C, H, W)
        x = self.fc(x) # -> (T, B, num_cls)

        return x.mean(0) # -> (B, num_cls)
    
class VGGEncoder(nn.Module):
    def __init__(self, args):
        super(VGGEncoder, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        conv = []
        in_channel = C

        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                conv.append(layer.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
            else:
                out_channel = x
                conv.append(
                    layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                conv.append(layer.BatchNorm2d(out_channel))
                conv.append(
                    neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU())
                in_channel = out_channel

        self.conv_layer = nn.Sequential(*conv)

        if int(layer_num) in [5, 9]:
            self.fc_layer = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 1024, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                # layer.Linear(1024, args.num_classes, bias=self.bias_flag),
            )
        else:
            self.fc_layer = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                layer.Linear(4096, 4096, bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU(),
                # layer.Linear(4096, args.num_classes, bias=self.bias_flag),
            )

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.conv_layer(x)  # (T, B, C, H, W)
        x = self.fc_layer(x) # -> (T, B, num_cls)

        return x # -> (T, B, D)
    

class VGGClassifier(nn.Module):
    def __init__(self, args, num_k=None):
        super(VGGClassifier, self).__init__()

        self.bias_flag = False if args.snn else True
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'

        hidden_dim = 1024 if int(layer_num) in [5, 9] else 4096
        if num_k is not None:
            hidden_dim = hidden_dim * num_k
        self.fc = layer.Linear(hidden_dim, args.num_classes, bias=self.bias_flag)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        x = self.fc(x)
        return x.mean(0) # -> (B, D)


class Staged_VGG(nn.Module):
    def __init__(self, args):
        ''' VGG '''
        super(Staged_VGG, self).__init__()

        C, H, W = INPUT_SIZE[args.dataset]
        self.bias_flag = False if args.snn else True
        self.rate = args.rate
        pattern = re.compile(r'\d+')
        layer_num = int(pattern.findall(args.model)[0])
        assert int(layer_num) in [5, 9, 11, 13, 16, 19], f'current layer settings of {args.model} not support!'
        self.num_stage = 0
        in_channel = C
        # num_stage = 0
        setattr(self,f"layer{self.num_stage}",[])
        for x in vgg_cfg[f'VGG{layer_num}']:
            if x == 'P':
                stage = getattr(self, f"layer{self.num_stage}")
                stage.append(layer.MaxPool2d(kernel_size=2))
                stage = nn.Sequential(*stage)
                setattr(self,f"layer{self.num_stage}",stage)
                # conv.append(layer.MaxPool2d(kernel_size=2))
                H //= 2
                W //= 2
                self.num_stage += 1
                setattr(self,f"layer{self.num_stage}",[])
            else:
                out_channel = int(x * self.rate)
                block = nn.Sequential(
                    layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag),
                    layer.BatchNorm2d(out_channel, track_running_stats=args.trs),
                    neuron.LIFNode(detach_reset=True) if args.snn else nn.ReLU()
                )
                in_channel = out_channel
                stage = getattr(self,f"layer{self.num_stage}")
                stage.append(block)
                # conv.append(
                #     layer.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias_flag))
                # conv.append(layer.BatchNorm2d(out_channel,track_running_stats=args.trs))
                # conv.append(
                #     neuron.LIFNode(detach_reset=True, surrogate_function=surrogate.ATan()) if args.snn else nn.ReLU())
                # in_channel = out_channel

        # self.features = nn.Sequential(*conv)

        if int(layer_num) in [5, 9]:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, int(1024*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True) if args.snn else nn.ReLU(),
                layer.Linear(int(1024*args.rate), args.num_classes, bias=self.bias_flag),
            )
        else:
            self.fc = nn.Sequential(
                layer.Flatten(),
                layer.Linear(in_channel * H * W, int(4096*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True) if args.snn else nn.ReLU(),
                layer.Linear(int(4096*args.rate), int(4096*args.rate), bias=self.bias_flag),
                neuron.LIFNode(detach_reset=True) if args.snn else nn.ReLU(),
                layer.Linear(int(4096*args.rate), args.num_classes, bias=self.bias_flag),
            )
        # print(f"rate:{self.rate},channel:{in_channel},H:{H},W:{W}")

        # for m in self.modules():
        #     if isinstance(m, (layer.Conv2d, layer.Linear)):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=2)

        functional.set_step_mode(self, 'm')

    def forward(self, x):
        # x = self.features(x)  # (T, B, C, H, W)
        for i in range(self.num_stage):
            stage = getattr(self,f"layer{i}")
            x = stage(x)
        x = self.fc(x) # -> (T, B, num_cls)
        return x.mean(0) # -> (B, num_cls)

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
    print(block_id)
    return block_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-model",type=str,default="vgg16")
    parser.add_argument("-dataset",type=str,default="cifar10")
    parser.add_argument("-snn",action="store_true")
    parser.add_argument("-rate",type=float,default=1.0)
    parser.add_argument("-num_classes", type=int, default=10)
    parser.add_argument("-trs",action="store_true")
    args = parser.parse_args()
    args.layer2_block =\
        [[0, 1, 3, 5, 7],
         [0, 2, 4, 6, 8],
         [0, 2, 4, 7, 10]]
    args.num_hidden_layers = [11,13,16]
    model = Staged_VGG(args)
    dummy_names = []
    leave_one_names = []
    res = 4
    dummy_names.append(
        [name for name, param in model.named_parameters() if param.requires_grad and "bias" not in name])
    leave_one_names.append(
        [name for name in dummy_names[0] if
         "fc" not in name and get_block_id(name, 2, args, type=name.split(".")[-1]) == args.num_hidden_layers[2] - res])
    print(leave_one_names)
    pass