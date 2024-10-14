import re
import copy
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate
import torch.nn.functional as F
from utils import INPUT_SIZE


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_snn=True,trs=False):
        super(Block, self).__init__()
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_channels, track_running_stats=trs)
        self.act1 = neuron.IFNode(detach_reset=True) if is_snn else nn.ReLU()
        self.conv2 = layer.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_channels * self.expansion, track_running_stats=trs)
        self.act2 = neuron.IFNode(detach_reset=True) if is_snn else nn.ReLU()
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels * self.expansion, kernel_size=3, stride=stride, padding=1,bias=False),
                layer.BatchNorm2d(out_channels * self.expansion,track_running_stats=trs),
            )

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act1(res)
        res = self.conv2(res)
        res = self.bn2(res)
        out = self.act2(res + self.shortcut(x))
        return out


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

        C, H, W = INPUT_SIZE[args.dataset]
        self.block = Block
        if args.has_rate:
            self.conv1 = nn.Sequential(
                layer.Conv2d(C, int(self.in_channels * args.rate), kernel_size=7, padding=3, stride=2),
                layer.BatchNorm2d(int(self.in_channels * args.rate),track_running_stats=args.trs),
                neuron.IFNode(detach_reset=True) if args.snn else nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(
                layer.Conv2d(C, self.in_channels, kernel_size=7, padding=3, stride=2),
                layer.BatchNorm2d(self.in_channels,track_running_stats=args.trs),
                neuron.IFNode(detach_reset=True) if args.snn else nn.ReLU()
            )
        # conv1.weight
        self.layer1 = self._make_layer(self.block, 64 * k, num_block[0], 2, args)
        self.layer2 = self._make_layer(self.block, 128 * k, num_block[1], 2, args)
        self.layer3 = self._make_layer(self.block, 256 * k, num_block[2], 2, args)
        self.layer4 = self._make_layer(self.block, 512 * k, num_block[3], 2, args)

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
            layers.append(block(self.in_channels, out_channels, s, args.snn,args.trs))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)

        return output
        # return output.mean(0)  # B, n_cls