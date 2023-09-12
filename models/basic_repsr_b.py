import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange,repeat
import argparse


class RepSR_MODEL(nn.Module):
    def __init__(self, params):
        super(RepSR_MODEL, self).__init__()
        self.image_mean = params.image_mean
        num_inputs = params.num_channels
        scale = params.scale
        kernel_size = 3
        self.scale = scale
        self.num_blocks = params.num_blocks
        self.backbone = None
        self.upsampler = None
        num_outputs = scale * scale * params.num_channels
        # num_outputs = 12

        conv = nn.Conv2d(
            in_channels=num_inputs,
            out_channels=params.num_residual_units,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        init.kaiming_normal_(conv.weight, mode='fan_in')
        if conv.bias is not None:
            init.constant_(conv.bias, 0)
        self.head = conv

        backbone = []
        # backbone += [Block(num_inputs, self.in_channels*2, self.in_channels)]
        for _ in range(self.num_blocks):
            backbone += [Block(params.num_residual_units, params.num_residual_units*2, params.num_residual_units)]

        # backbone += [Block(params.num_residual_units, params.num_residual_units*2, num_outputs)]

        conv = nn.Conv2d(
            in_channels=params.num_residual_units,
            out_channels=num_outputs,
            kernel_size=kernel_size,
            padding=kernel_size // 2)
        init.kaiming_normal_(conv.weight, mode='fan_out')
        init.zeros_(conv.bias)
        self.tail = conv

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x = x - self.image_mean
        out = self.head(x)
        out = self.backbone(out)
        out = self.tail(out)
        '''这里存在一个问题，跟一开始的输入一样，最后需要将feature的通道数改编成 num_outputs,
        但是由于每个block中都有 identity，所以最后一个block需要额外处理'''
        out = self.upsampler(out) + self.image_mean
        return out


class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 ):
        super(Block, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.conv1_3x3 = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3,
                                   padding=1)

        self.conv2_3x3 = nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3,
                                   padding=1)

        self.bn1 = nn.BatchNorm2d(self.mid_channels)
        self.bn2 = nn.BatchNorm2d(self.mid_channels)

        self.conv1_1x1 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1,
                                   padding=0)

        self.conv2_1x1 = nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1,
                                   padding=0)

        self.init_weights()

    def forward(self, x):
        if self.training:
            out1 = self.conv1_1x1(self.bn1(self.conv1_3x3(x)))
            out2 = self.conv2_1x1(self.bn2(self.conv2_3x3(x)))
            out = out1 + out2 + x
        else:
            out = self.repsrb_reparam(x)
        return out

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1.0)
                init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    # 在fuse_bn中，可以直接传入conv和bn，也就是说我们在调用时，直接将3x3的卷积作为参数传递
    # def fuse_bn(self, conv, bn):
    #     param_size = conv.weight.shape
    #     conv_w = conv.weight
    #     if conv.bias is not None:
    #         conv_b = conv.bias
    #     else:
    #         conv_b = torch.zeros(param_size[0])
    #     gamma = bn.weight
    #     beta = bn.bias
    #     mean = bn.running_mean
    #     var = bn.running_var
    #     eps = bn.eps
    #
    #     std = (var + eps).sqrt()
    #     t = (gamma / std).reshape(-1, 1, 1, 1)
    #
    #     weight = conv_w * t
    #     bias = beta + gamma * (conv_b - mean) / std
    #
    #     return weight, bias
    def _fusebn(self, conv, bn):
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        weight = torch.mm(w_bn, w_conv).view(conv.weight.size())

        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))

        b_conv = torch.mm(w_bn, b_conv.view(conv.out_channels, -1)).squeeze()

        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps)
        )
        bias = torch.matmul(w_bn, b_conv) + b_bn

        return weight, bias

    def _equivalent_conv(self):
        self.repsrb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True
        )

        c3w_1, c3b_1 = self.fuse_bn(self.conv1_3x3, self.bn1)
        c1w_1 = self.conv1_1x1.weight
        c1b_1 = self.conv1_1x1.bias

        # 改变两个卷积核权重的轴顺序
        _c3w_1 = rearrange(c3w_1, 'c_mid c_in h w -> (h w) c_mid c_in')
        _c1w_1 = rearrange(c1w_1, 'c_out c_mid h w -> (h w) c_out c_mid')

        # merged_weight_1 = torch.bmm(_c1w_1.expand(3*3, self.out_channels, self.mid_channels), _c3w_1)
        # 为什么采用repeat函数得出的结果会更好一点
        merged_weight_1 = torch.bmm(repeat(_c1w_1, '1 c_out c_mid -> 9 c_out c_mid'), _c3w_1)
        merged_weight_1 = rearrange(merged_weight_1, '(h w) c_out c_in -> c_out c_in h w', h=3)

        # 改变两个卷积核偏置的轴顺序
        _c3b_1 = c3b_1.reshape(self.mid_channels, 1)
        _c1w_1 = _c1w_1.reshape(self.out_channels, self.mid_channels)

        merged_bias_1 = torch.mm(_c1w_1, _c3b_1).view(-1, ) + c1b_1

        c3w_2, c3b_2 = self.fuse_bn(self.conv2_3x3, self.bn2)
        c1w_2 = self.conv2_1x1.weight
        c1b_2 = self.conv2_1x1.bias

        # 改变两个卷积核权重的轴顺序
        _c3w_2 = rearrange(c3w_2, 'c_mid c_in h w -> (h w) c_mid c_in')
        _c1w_2 = rearrange(c1w_2, 'c_out c_mid h w -> (h w) c_out c_mid')

        # merged_weight_2 = torch.bmm(_c1w_2.expand(3*3, self.out_channels, self.mid_channels), _c3w_2)
        merged_weight_2 = torch.bmm(repeat(_c1w_2, '1 c_out c_mid -> 9 c_out c_mid'), _c3w_2)
        merged_weight_2 = rearrange(merged_weight_2, '(h w) c_out c_in -> c_out c_in h w', h=3)

        # 改变两个卷积核偏置的轴顺序
        _c3b_2 = c3b_2.reshape(self.mid_channels, 1)
        _c1w_2 = _c1w_2.reshape(self.out_channels, self.mid_channels)

        merged_bias_2 = torch.mm(_c1w_2, _c3b_2).view(-1, ) + c1b_2

        merged_weight = merged_weight_1 + merged_weight_2
        merged_bias = merged_bias_1 + merged_bias_2
        # merge residual path
        for i in range(self.out_channels):
            merged_weight[i, i, 1, 1] += 1

        self.repsrb_reparam.weight = nn.Parameter(merged_weight, requires_grad=False)
        self.repsrb_reparam.bias = nn.Parameter(merged_bias, requires_grad=False)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_mean', help='image_mean', default=0.5, type=float)
#     parser.add_argument('--num_residual_units', help='Number of residual units of networks',
#                         default=24, type=int)
#     parser.add_argument('--num_blocks', help='number of residual blocks', default=16, type=int)
#     parser.add_argument('--num_channels', help='number of input channels', default=3,
#                         type=int)
#     parser.add_argument('--scale', help='scale of res', default=2, type=int)
#
#     params = parser.parse_args()
#     model = RepSR_MODEL(params)
#     module = model.module if hasattr(model, 'module') else model
#
#     print(module)
