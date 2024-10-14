import re
import copy
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate
import torch.nn.functional as F
from utils import INPUT_SIZE
from models.raw_resnet import mem_update
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants



# class BasicBlock_18(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             neuron.LIFNode(tau=4.0, detach_reset=True, surrogate_function=ActFun.apply),
#             Snn_Conv2d(in_channels,
#                        out_channels,
#                        kernel_size=3,
#                        stride=stride,
#                        padding=1,
#                        bias=False),
#             layer.BatchNorm2d(out_channels),
#             neuron.LIFNode(tau=4.0, detach_reset=True, surrogate_function=ActFun.apply),
#             Snn_Conv2d(out_channels,
#                        out_channels * BasicBlock_18.expansion,
#                        kernel_size=3,
#                        padding=1,
#                        bias=False),
#             layer.BatchNorm2d(out_channels * BasicBlock_18.expansion),
#         )
#         self.shortcut = nn.Sequential()
#
#         if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 Snn_Conv2d(in_channels,
#                            out_channels * BasicBlock_18.expansion,
#                            kernel_size=1,
#                            stride=stride,
#                            bias=False),
#                 layer.BatchNorm2d(out_channels * BasicBlock_18.expansion),
#             )
#
#     def forward(self, x):
#         return self.residual_function(x) + self.shortcut(x)


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class Scale_Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride,is_snn,rate,trs):
        super(Scale_Block, self).__init__()
        n1 = layer.BatchNorm2d(out_channels,track_running_stats=trs)
        n2 = layer.BatchNorm2d(out_channels,track_running_stats=trs)
        self.act1 = neuron.LIFNode(tau=4.0, detach_reset=True, surrogate_function=surrogate.Sigmoid()) if is_snn else nn.ReLU()
        self.act2 = neuron.LIFNode(tau=4.0, detach_reset=True, surrogate_function=surrogate.Sigmoid()) if is_snn else nn.ReLU()
        self.conv1 = layer.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = n1
        self.conv2 = layer.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = n2
        self.scaler = Scaler(rate)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # layer.MaxPool2d(2, 2),
                layer.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels * self.expansion,track_running_stats=trs),
            )

    def forward(self, x):
        out = self.act1(x)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.n1(self.conv1(self.scaler(out)))
        out = self.act2(out)
        out = self.n2(self.conv2(self.scaler(out)))
        out += shortcut
        return out


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_snn=True,trs=False):
        super(Block, self).__init__()
        self.residual_function = nn.Sequential(
            # mem_update(),
            neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.Sigmoid()) if is_snn else nn.ReLU(),
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(out_channels,track_running_stats=trs),
            neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.Sigmoid()) if is_snn else nn.ReLU(),
            # mem_update(),
            layer.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(out_channels * self.expansion,track_running_stats=trs)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # layer.MaxPool2d(2, 2),
                layer.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels * self.expansion,track_running_stats=trs),
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class Resnet(nn.Module):
    def __init__(self, args):
        super(Resnet, self).__init__()
        k = 1
        self.in_channels = 64 * k

        self.num_classes = args.num_classes

        pattern = re.compile(r'\d+')
        depths = int(pattern.findall(args.model)[0])
        if depths == 18:
            num_block = [2, 2, 2, 2]
        elif depths == 14:
            num_block = [1,1,2,2]
        elif depths == 22:
            num_block = [2,2,3,3]
        elif depths == 26:
            num_block = [3,3,3,3]
        elif depths == 10:
            num_block = [1, 1, 1, 1]
        elif depths == 34:
            num_block = [3, 4, 6, 3]
        else:
            raise NotImplementedError(f'Invalid model {args.model}, only support `resnet10`, `resnet18` and `resnet34`')
        self.num_block = num_block
        C, H, W = INPUT_SIZE[args.dataset]
        self.block = Scale_Block if args.has_rate else Block
        if args.has_rate:
            self.conv1 = nn.Sequential(
                layer.Conv2d(C, int(self.in_channels * args.rate), kernel_size=7, padding=3, stride=2),
                layer.BatchNorm2d(int(self.in_channels * args.rate),track_running_stats=args.trs),
            )
        else:
            self.conv1 = nn.Sequential(
                layer.Conv2d(C, self.in_channels, kernel_size=7, padding=3, stride=2),
                layer.BatchNorm2d(self.in_channels,track_running_stats=args.trs),
            )

        # conv1.weight
        self.layer1 = self._make_layer(self.block, 64 * k, num_block[0], 2, args)
        self.layer2 = self._make_layer(self.block, 128 * k, num_block[1], 2, args)
        self.layer3 = self._make_layer(self.block, 256 * k, num_block[2], 2, args)
        self.layer4 = self._make_layer(self.block, 512 * k, num_block[3], 2, args)
        self.lif = neuron.LIFNode(tau=4., v_threshold=0.5, detach_reset=True, surrogate_function=surrogate.Sigmoid()) if args.snn else nn.ReLU()
        # self.lif = mem_update()
        # self.pool = layer.AdaptiveAvgPool2d((1, 1))
        if args.has_rate:
            self.fc = layer.Linear(int(512 * self.block.expansion * k * args.rate), self.num_classes)
        else:
            self.fc = layer.Linear(512 * self.block.expansion * k, self.num_classes)
        functional.set_step_mode(self, 'm')

    def _make_layer(self, block, out_channels, num_blocks, stride, args):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if args.has_rate:
                layers.append(block(int(self.in_channels * args.rate), int(out_channels * args.rate), s, args.snn,args.rate,args.trs))
                self.in_channels = out_channels * block.expansion
            else:
                layers.append(block(self.in_channels, out_channels, s, args.snn,args.trs))
                self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.lif(output)

        # output = self.pool(output)

        # output = F.adaptive_max_pool3d(output,(None,1,1))
        output = F.adaptive_avg_pool3d(output,(None,1,1))
        output = output.view(output.size()[0], output.size()[1], -1)

        output = output.sum(dim=0) / output.size()[0]
        sequence_0_1 = output
        output = self.fc(output)

        return output, sequence_0_1
        # return output.mean(0)  # B, n_cls

    # ssh - p
    # 19325
    # root @ connect.westc.gpuhub.com
    # BCBUtOv1R0I7

if __name__ == "__main__":
    from energy_sim.forward_graph_parser import energy_count
    import argparse
    parser = argparse.ArgumentParser("test")
    parser.add_argument("-num_classes", type=int, default=10)
    parser.add_argument("-dataset", type=str, default="cifar10")
    parser.add_argument("-has_rate", type=bool, default=True)
    parser.add_argument("-rate", type=float, default=0.25)
    parser.add_argument("-model", type=str, default="resnet10")
    parser.add_argument("-trs", type=bool, default=True)
    args = parser.parse_args()
    args.snn = True
    model = Resnet(args)

    # model_size = sum(p.numel() for p in model.parameters()) / 1e6
    # print("Num of parameters for model = {}".format(model_size))