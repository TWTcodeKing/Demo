import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import INPUT_SIZE
import numpy as np
import copy


thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.25  # decay constants
# num_classes = 1000
# time_window = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class mem_update(nn.Module):

    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        time_window = x.shape[0]
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1 - spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    """TDBN"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features,track_running_stats=False)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class batch_norm_2d1(nn.Module):
    """TDBN-Zero init"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features,track_running_stats=False)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0)
            nn.init.zeros_(self.bias)


class Snn_Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 marker='b'):
        super(Snn_Conv2d,
              self).__init__(in_channels, out_channels, kernel_size, stride,
                             padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        time_window = input.shape[0]
        weight = self.weight
        h = (input.size()[3] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        w = (input.size()[4] - self.kernel_size[0] +
             2 * self.padding[0]) // self.stride[0] + 1
        c1 = torch.zeros(time_window,
                         input.size()[1],
                         self.out_channels,
                         h,
                         w,
                         device=input.device)
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        return c1


def layer_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Snn_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def layer_conv1x1(in_planes, out_planes, stride=1):
    return Snn_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

        bn_list = [batch_norm_2d(in_planes) for _ in range(num_conv_layers)]
        relu_list = [mem_update() if is_snn else nn.ReLU() for _ in range(num_conv_layers)]
        # avg_pool = layer.AdaptiveAvgPool2d((1, 1))
        # flatten = layer.Flatten()

        layers = []
        for i in range(num_conv_layers):
            layers.append(conv_list[i])
            layers.append(scaler)
            layers.append(bn_list[i])
            layers.append(relu_list[i])
        # layers.append(avg_pool)
        # layers.append(flatten)

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, inp, pred=None):
        output = self.layers(inp)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0) / output.size()[0]
        output = self.fc(output)
        return output


class BasicBlock_18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, is_snn=True, rate=1.0):
        super().__init__()
        self.act1 = mem_update() if is_snn else nn.ReLU()
        self.scaler = Scaler(rate)
        self.conv1 = Snn_Conv2d(in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False)
        self.bn1 = batch_norm_2d(out_channels)
        self.act2 = mem_update() if is_snn else nn.ReLU()
        self.conv2 = Snn_Conv2d(out_channels,
                     out_channels * BasicBlock_18.expansion,
                     kernel_size=3,
                     padding=1,
                     bias=False)
        self.bn2 = batch_norm_2d1(out_channels * BasicBlock_18.expansion)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock_18.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels,
                           out_channels * BasicBlock_18.expansion,
                           kernel_size=1,
                           stride=stride,
                           bias=False),
                batch_norm_2d(out_channels * BasicBlock_18.expansion),
            )

    def forward(self, x):
        out = self.act1(x)
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = self.bn1(self.conv1(self.scaler(out)))
        out = self.act2(out)
        out = self.bn2(self.conv2(self.scaler(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, args,ee_layer_locations=[], scale=1.):
        super(ResNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        factor = 1

        self.scale = scale
        self.in_planes = int(64 * scale * factor)
        self.num_blocks = len(ee_layer_locations) + 1
        self.num_classes = num_classes
        self.snn = args.snn
        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        ee_block_list = []
        ee_layer_list = []

        for ee_layer_idx in ee_layer_locations:
            b, l = self.find_ee_block_and_layer(layers, ee_layer_idx)
            ee_block_list.append(b)
            ee_layer_list.append(l)
        C, H, W = INPUT_SIZE[args.dataset]
        # if self.num_classes > 100:
        #     self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=5, stride=2, padding=3, bias=False)
        self.conv1 = nn.Sequential(
            Snn_Conv2d(C,
                       self.in_planes,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            batch_norm_2d(self.in_planes),
        )

        layer1, ee1 = self._make_layer(block, int(64 * scale * factor), layers[0], stride=1,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 0])
        layer2, ee2 = self._make_layer(block, int(128 * scale * factor), layers[1], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 1])
        layer3, ee3 = self._make_layer(block, int(256 * scale * factor), layers[2], stride=2,
                                       ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 2])
        self.layers = nn.ModuleList([layer1, layer2, layer3])
        self.ee_classifiers = nn.ModuleList([ee1, ee2, ee3])
        # self.pool = layer.AdaptiveAvgPool2d((1, 1))
        if self.num_classes > 1:
            layer4, ee4 = self._make_layer(block, int(512 * scale * factor), layers[3], stride=2,
                                           ee_layer_locations=[l for i, l in enumerate(ee_layer_list) if ee_block_list[i] == 3])
            self.layers.append(layer4)
            self.ee_classifiers.append(ee4)

            num_planes = int(512 * scale) * block.expansion
            self.linear = nn.Linear(num_planes, num_classes)
        else:
            num_planes = int(256 * scale) * block.expansion
            self.linear = nn.Linear(num_planes, num_classes)
        # functional.set_step_mode(self, 'm')
        # else:
        #     num_planes = int(64 * scale) * block.expansion
        #     self.linear = nn.Linear(num_planes, num_classes)

    def _make_layer(self, block_type, planes, num_block, stride, ee_layer_locations):
        strides = [stride] + [1] * (num_block - 1)

        ee_layer_locations_ = ee_layer_locations + [num_block]
        layers = [[] for _ in range(len(ee_layer_locations_))]

        ee_classifiers = []

        if len(ee_layer_locations_) > 1:
            start_layer = 0
            counter = 0
            for i, ee_layer_idx in enumerate(ee_layer_locations_):

                for _ in range(start_layer, ee_layer_idx):
                    layers[i].append(block_type(self.in_planes, planes, strides[counter], is_snn=self.snn,rate=self.scale))
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
                layers[0].append(block_type(self.in_planes, planes, strides[i], rate=self.scale,is_snn=self.snn))
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
            if final_out is not None:
                if manual_early_exit_index > sum([len(ee) for ee in self.ee_classifiers[:counter+1]]):
                    manual_early_exit_index_ = 0
                elif manual_early_exit_index:
                    manual_early_exit_index_ = manual_early_exit_index - sum([len(ee) for ee in self.ee_classifiers[:counter]])
                else:
                    manual_early_exit_index_ = manual_early_exit_index

                final_out = self._block_forward(self.layers[counter], self.ee_classifiers[counter], final_out, ee_outs, manual_early_exit_index_)
            counter += 1

        preds = ee_outs

        if final_out is not None:
            # out = self.pool(final_out)
            # out = torch.flatten(out,2)
            output = F.adaptive_avg_pool3d(final_out, (None, 1, 1))
            output = output.view(output.size()[0], output.size()[1], -1)
            output = output.sum(dim=0) / output.size()[0]
            out = self.linear(output)
            preds.append(out)

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
    return ResNet(BasicBlock_18, [3, 3, 3, 3], args.num_classes, scale=args.rate,args=args)
    # no ee layer


def resnet34_4(args):
    ee_layer_locations = args.ee_layer_locations
    return ResNet(BasicBlock_18, [3, 4, 6, 3], args.num_classes, ee_layer_locations=ee_layer_locations,args=args,
                  scale=args.rate)
    # has ee layer
