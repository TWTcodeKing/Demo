import re

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron

from utils import INPUT_SIZE


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output


class SimpleSPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, embed_dims=256,rate=1.0,trs=True):
        super().__init__()
        
        self.is_snn = True
        self.image_size = [img_size_h, img_size_w]
        embed_dims = int(embed_dims * rate)
        self.proj_conv = layer.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = layer.BatchNorm2d(embed_dims // 2,track_running_stats=trs)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()
        self.maxpool1 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = layer.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims,track_running_stats=trs)
        self.proj_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) 

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = layer.BatchNorm2d(embed_dims,track_running_stats=trs)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()
        self.scale = Scaler(rate)
        functional.set_step_mode(self,"m")

    def forward(self, x):
        T, B, _, H, W = x.shape

        x = self.proj_conv(self.scale(x))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif(x).contiguous()
        x = self.maxpool1(x)

        x = self.proj_conv2(self.scale(x))
        x = self.proj_bn2(x).reshape(T, B, -1, H // 2, W // 2).contiguous() 
        x = self.proj_lif2(x).contiguous() 
        x = self.maxpool2(x)
        # x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x_feat = x
        x = self.rpe_conv(self.scale(x))
        # x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_bn(x)
        x = self.rpe_lif(x) 
        x = x + x_feat 
        print(x.shape)
        return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, in_channels=2, embed_dims=256,trs=True):
        super().__init__()
        
        self.is_snn = True

        self.image_size = [img_size_h, img_size_w]

        self.proj_conv = layer.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = layer.BatchNorm2d(embed_dims // 8,track_running_stats=trs)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()

        self.proj_conv1 = layer.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = layer.BatchNorm2d(embed_dims // 4,track_running_stats=trs)
        self.proj_lif1 = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()

        self.proj_conv2 = layer.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = layer.BatchNorm2d(embed_dims // 2,track_running_stats=trs)
        self.proj_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()
        self.maxpool2 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = layer.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = layer.BatchNorm2d(embed_dims,track_running_stats=trs)
        self.proj_lif3 = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()
        self.maxpool3 = layer.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False) 

        self.rpe_conv = layer.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = layer.BatchNorm2d(embed_dims,track_running_stats=trs)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if self.is_snn else nn.ReLU()

    def forward(self, x):
        T, B, _, H, W = x.shape

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif(x).contiguous()

        x = self.proj_conv1(x) 
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif1(x).contiguous()

        x = self.proj_conv2(x) 
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous() 
        x = self.proj_lif2(x).contiguous() 
        x = self.maxpool2(x) 

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous() 
        x = self.proj_lif3(x).contiguous() 
        x = self.maxpool3(x) 
        
        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous() 
        x = self.rpe_conv(x) 
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous() 
        x = self.rpe_lif(x) 
        x = x + x_feat 

        return x


class SMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, is_snn=True,rate=1.,trs=True):
        super().__init__()

        hidden_features = hidden_features or in_features

        self.fc1 = layer.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.bn1 = layer.BatchNorm1d(hidden_features,track_running_stats=trs)
        self.lif1 = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()

        self.fc2 = layer.Conv1d(hidden_features, in_features, kernel_size=1, stride=1)
        self.bn2 = layer.BatchNorm1d(in_features,track_running_stats=trs)
        self.lif2 = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()

        self.c_hidden = hidden_features
        self.scale = Scaler(rate)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3) # -> T, B, C, HW
        x = self.lif1(x)
        x = self.scale(self.fc1(x))
        x = self.bn1(x).reshape(T, B, self.c_hidden, H * W).contiguous()

        x = self.lif2(x)
        x = self.scale(self.fc2(x))
        x = self.bn2(x).reshape(T, B, C, H, W).contiguous()

        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, is_snn=True, rate=1.,trs=True) -> None:
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.scale = 0.125

        self.head_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()

        self.q_conv = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False), 
            layer.BatchNorm2d(dim,track_running_stats=trs)
        )
        self.k_conv = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False), 
            layer.BatchNorm2d(dim,track_running_stats=trs)
        )
        self.v_conv = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False), 
            layer.BatchNorm2d(dim,track_running_stats=trs)
        )

        self.q_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()
        self.k_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()
        self.v_lif = neuron.LIFNode(tau=2.0, detach_reset=True) if is_snn else nn.ReLU()

        self.attn_lif = neuron.LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True) if is_snn else nn.ReLU()

        self.proj_conv = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False), 
            layer.BatchNorm2d(dim,track_running_stats=trs)
        )
        self.scaler = Scaler(rate)

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.scaler(self.q_conv(x))
        k = self.scaler(self.k_conv(x))
        v = self.scaler(self.v_conv(x))

        q = self.q_lif(q).flatten(3) # -> T, B, C, HW
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_lif(k).flatten(3)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.v_lif(v).flatten(3)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)

        x = self.scaler(self.proj_conv(x))

        return x
    
class SBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., is_snn=True,rate=1.,trs=True):
        super().__init__()

        self.is_snn = is_snn
        dim = int(rate * dim)
        self.attn = SSA(dim, num_heads, self.is_snn,rate=1.,trs=trs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SMLP(dim, mlp_hidden_dim, self.is_snn,trs=trs)
        functional.set_step_mode(self,"m")
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x
        
class SViT(nn.Module):
    ''' Meta-Spikeformer '''
    def __init__(self, args):
        super().__init__()

        self.num_classes = args.num_classes
        self.rate = args.rate
        pattern = re.compile(r'\d+')
        depths = int(pattern.findall(args.model)[0])
        self.depths = depths

        C, H, W = INPUT_SIZE[args.dataset]

        self.patch_embed = SimpleSPS(H, W, C, args.hidden_dim,args.rate,trs=args.trs) # SPS(H, W, C, args.hidden_dim)

        self.block = nn.ModuleList([SBlock(args.hidden_dim, args.num_heads, is_snn=args.snn,rate=args.rate,trs=args.trs) for _ in range(self.depths)])

        self.lif = neuron.LIFNode(tau=2.0, detach_reset=True) if args.snn else nn.ReLU()

        # classification head
        self.head = layer.Linear(int(args.hidden_dim*args.rate), self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        functional.set_step_mode(self, 'm')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x) # T, B, C, H, W

        x = self.lif(x.flatten(3).mean(3))
        sequence_0_1 = x
        print(sequence_0_1.shape)
        x = self.head(x)

        return x.mean(0) , sequence_0_1# B, cls


if __name__ == "__main__":
    block = SimpleSPS(img_size_h=34,img_size_w=34,in_channels=2,embed_dims=384)
    input_tensor = torch.randn(4,1,2,34,34)
    output_tensor = block(input_tensor)
    attn_block = SBlock(dim=384,num_heads=8)
    output_tensor = attn_block(output_tensor)
    output_tensor = output_tensor.flatten(3).mean(3)
    print(output_tensor.shape)