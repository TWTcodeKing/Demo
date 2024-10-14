import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from utils import INPUT_SIZE
import numpy as np
import copy


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


def layer_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def layer_conv1x1(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes, num_conv_layers=3, reduction=1, scale=1.,is_snn=True):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_conv_layers = num_conv_layers
        self.reduction = reduction
        self.scale = scale

        if scale < 1:
            scaler = Scaler(scale)
        else:
            scaler = nn.Identity()

        if reduction == 1:
            conv_list = [layer_conv3x3(in_planes,in_planes) for _ in range(num_conv_layers)]
        else:
            conv_list = [layer_conv3x3(in_planes, int(in_planes/reduction))]
            in_planes = int(in_planes/reduction)
            conv_list.extend([layer_conv3x3(in_planes, in_planes) for _ in range(num_conv_layers-1)])

        bn_list = [layer.BatchNorm2d(in_planes, track_running_stats=False) for _ in range(num_conv_layers)]
        relu_list = [neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan()) \
                     if is_snn else nn.ReLU() for _ in range(num_conv_layers)]
        avg_pool = layer.AdaptiveAvgPool2d((1, 1))
        flatten = layer.Flatten()

        layers = []
        for i in range(num_conv_layers):
            layers.append(conv_list[i])
            layers.append(scaler)
            layers.append(bn_list[i])
            layers.append(relu_list[i])
        layers.append(avg_pool)
        layers.append(flatten)

        self.layers = nn.Sequential(*layers)
        self.fc = layer.Linear(in_planes, num_classes)
        functional.set_step_mode(self,"m")

    def forward(self, inp, pred=None):
        output = self.layers(inp)
        output = self.fc(output)
        return output.mean(0)


class Scale_Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride,is_snn=True,trs=False,scale=1.0):
        super(Scale_Block, self).__init__()
        n1 = layer.BatchNorm2d(out_channels, momentum=None, track_running_stats=trs)
        n2 = layer.BatchNorm2d(out_channels, momentum=None, track_running_stats=trs)
        self.act1 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan()) if is_snn else nn.ReLU()
        self.act2 = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan()) if is_snn else nn.ReLU()
        self.conv1 = layer.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = n1
        self.conv2 = layer.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = n2
        self.scaler = Scaler(scale)

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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, args,ee_layer_locations=[], scale=1., trs=False):
        super(ResNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        factor = 1

        self.scale = scale
        self.in_planes = int(64 * scale * factor)
        self.num_blocks = len(ee_layer_locations) + 1
        self.num_classes = num_classes
        self.trs = trs
        self.snn = args.snn
        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        ee_block_list = []
        ee_layer_list = []
        # print("ee layer locations:",ee_layer_locations)
        # ee_layer_locations = []
        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)
        C, H, W = INPUT_SIZE[args.dataset]
        # if self.num_classes > 100:
        #     self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)
        self.conv1 = nn.Sequential(
            layer.Conv2d(C, self.in_planes, kernel_size=7, padding=3, stride=2),
            self.scaler,
            layer.BatchNorm2d(self.in_planes, track_running_stats=False),
        )

        layer1, ee1 = self._make_layer(block, int(64 * scale * factor), layers[0], stride=1,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0])
        layer2, ee2 = self._make_layer(block, int(128 * scale * factor), layers[1], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1])
        layer3, ee3 = self._make_layer(block, int(256 * scale * factor), layers[2], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2])
        self.layers = nn.ModuleList([layer1, layer2, layer3])
        self.ee_classifiers = nn.ModuleList([ee1, ee2, ee3])
        self.pool = layer.AdaptiveAvgPool2d((1, 1))
        if self.num_classes > 1:
            layer4, ee4 = self._make_layer(block, int(512 * scale * factor), layers[3], stride=2,
                                           ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 3])
            self.layers.append(layer4)
            self.ee_classifiers.append(ee4)

            num_planes = int(512 * scale) * block.expansion
            self.linear = layer.Linear(num_planes, num_classes)
        else:
            num_planes = int(256 * scale) * block.expansion
            self.linear = layer.Linear(num_planes, num_classes)
        functional.set_step_mode(self, 'm')
        # else:
        #     num_planes = int(64 * scale) * block.expansion
        #     self.linear = nn.Linear(num_planes, num_classes)

    def _make_layer(self, block_type, planes, num_block, stride, ee_layer_locations):
        strides = [stride] + [1] * (num_block - 1)

        ee_layer_locations_ = ee_layer_locations + [num_block]   # [3,7,15]   [3,7,15,2]
        layers = [[] for _ in range(len(ee_layer_locations_))]

        ee_classifiers = []

        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):

                for _ in range(start_layer, ee_layer_idx):
                    layers[i].append(block_type(self.in_planes, planes, strides[counter], is_snn=self.snn,trs=self.trs, scale=self.scale))
                    self.in_planes = planes * block_type.expansion
                    counter += 1
                start_layer = ee_layer_idx

                if ee_layer_idx == 0:
                    num_planes = self.in_planes
                else:
                    num_planes = planes * block_type.expansion

                if i < len(ee_layer_locations_) - 1:
                    ee_classifiers.append(Classifier(num_planes, num_classes=self.num_classes,
                                                     reduction=block_type.expansion, scale=self.scale,
                                                     is_snn=self.snn))

        else:
            for i in range(num_block):
                layers[0].append(block_type(self.in_planes, planes, strides[i], trs=self.trs, scale=self.scale,is_snn=self.snn))
                self.in_planes = planes * block_type.expansion

        return nn.ModuleList([nn.Sequential(*l) for l in layers]), nn.ModuleList(ee_classifiers)

    @staticmethod
    def find_ee_block_and_layer(layers, layer_idx):
        temp_array = np.zeros((sum(layers)), dtype=int)
        cum_array = np.cumsum(layers)
        for i in range(1, len(cum_array)):
            temp_array[cum_array[i-1]:] += 1
        block = temp_array[layer_idx]
        if block == 0:
            layer = layer_idx
        else:
            layer = layer_idx - cum_array[block-1]
        return block, layer

    def forward(self, x, manual_early_exit_index=0):
        # final_out = F.relu(self.bn1(self.scaler(self.conv1(x))))
        # if self.num_classes > 100:
        #     final_out = F.max_pool2d(final_out, kernel_size=3, stride=2, padding=1)
        final_out = self.conv1(x)
        ee_outs = []
        counter = 0

        while counter < len(self.layers):
            # print(manual_early_exit_index)
            if final_out is not None:
                if manual_early_exit_index > sum([len(ee) for ee in self.ee_classifiers[:counter+1]]):
                    manual_early_exit_index_ = 0
                elif manual_early_exit_index:
                    manual_early_exit_index_ = manual_early_exit_index - sum([len(ee) for ee in self.ee_classifiers[:counter]])
                else:
                    manual_early_exit_index_ = manual_early_exit_index
                final_out = self._block_forward(self.layers[counter], self.ee_classifiers[counter], final_out, ee_outs, manual_early_exit_index_)
            counter += 1

        if manual_early_exit_index:
            preds = ee_outs
        else:
            preds = []
        if final_out is not None:
            out = self.pool(final_out)
            out = torch.flatten(out,2)
            out = self.linear(out)
            preds.append(out.mean(0))

        if manual_early_exit_index:
            assert len(preds) == manual_early_exit_index
        return preds

    def _block_forward(self, layers, ee_classifiers, x, outs, early_exit=0):
        for i in range(len(layers)-1):
            x = layers[i](x)
            if outs:
                outs.append(ee_classifiers[i](x, outs[-1]))
            else:
                outs.append(ee_classifiers[i](x))
            if early_exit == i + 1:
                break
        if early_exit == 0:
            final_out = layers[-1](x)
        else:
            final_out = None
        return final_out


def resnet34_1(args):
    return ResNet(Scale_Block, [3, 3, 3, 3], args.num_classes, scale=args.rate,args=args)
    # no ee layer


def resnet34_4(args):
    ee_layer_locations = args.ee_layer_locations
    return ResNet(Scale_Block, [3, 4, 6, 3], args.num_classes, ee_layer_locations=ee_layer_locations,args=args,
                  scale=args.rate,trs=args.trs)
    # has ee layer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("test")
    parser.add_argument("-num_classes", type=int, default=10)
    parser.add_argument("-dataset", type=str, default="cifar10")
    parser.add_argument("-has_rate", type=bool, default=False)
    parser.add_argument("-model", type=str, default="resnet34")
    parser.add_argument("-trs", type=bool, default=True)
    args = parser.parse_args()
    args.snn = True
    args.ee_layer_locations = [3,7,15]
    args.rate = 1.0
    model = resnet34_4(args)
    x = torch.randn(4,1,3,32,32)
    y = model(x,manual_early_exit_index=3)
    print(len(y))
    pass