from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.ops import BinaryConv2d, rounding
# from basic_repsr import RepSR_Block

try:
    from speed_models import BlockBSpeedEstimator
except ImportError:
    pass

from collections import namedtuple

ModelOutput = namedtuple(
    'ModelOutput',
    'sr speed_accu speed_curr'
)
# 创建一个命名元组，其名称为 ModelOutput，其中包含三个参数，
# 可以使用output = Model Output(sr=*, speed_accu=*, speed_curr=*) 来实例化一个命名元组
# 实例化后，output可以像字典那样采用 output.sr 来查看当前值

__all__ = ['RepNAS_MODEL', ]

class RepNAS_MODEL(nn.Module):
    def __init__(self, params):
        super(RepNAS_MODEL, self).__init__()
        # datasets 中可以找到该自定义参宿和
        self.image_mean = params.image_mean
        multiplier = 2.0
        kernel_size = 3

        weight_norm = torch.nn.utils.weight_norm
        # 在datasets中，如 div2k 当中有定义
        num_inputs = params.num_channels
        # scale 在 _isr.py 中有定义，表示超分的倍率
        scale = params.scale
        self.scale = scale
        self.num_residual_units = params.num_residual_units
        self.num_blocks = params.num_blocks
        self.width_search = params.width_search

        num_outputs = scale * scale * params.num_channels

        # 由于在 Block 当中只设定了每个 repsr_block 的部分，网络框架的头部和尾部还需要另外定义

        conv = nn.Conv2d(
            in_channels=num_inputs,
            out_channels=params.num_residual_units,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )

        init.kaiming_normal_(conv.weight, mode='fan_in')
        if conv.bias is not None:
            init.constant_(conv.bias, 0)
        self.head = conv

        self.speed_estimator = BlockBSpeedEstimator('mask' if params.width_search else 'channel').eval()

        body = nn.ModuleList()
        for _ in range(params.num_blocks):
            body.append(AggregationLayer(
                num_residual_units=params.num_residual_units,
                kernel_size=kernel_size,
                weight_norm=weight_norm,
                width_search=params.width_search))
        self.body = body

        if self.width_search:
            self.mask = BinaryConv2d(in_channels=params.num_residual_units,
                                     out_channels=params.num_residual_units,
                                     groups=params.num_residual_units)

        conv = nn.Conv2d(
            in_channels=params.num_residual_units,
            out_channels=num_outputs,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.tail = conv

        shuf = []
        if scale > 1:
            shuf.append(nn.PixelShuffle(scale))
        self.shuf = nn.Sequential(*shuf)

        if params.pretrained:
            self.load_pretrained()

        # self.init_weights()


    def forward(self, x):
        x = x - self.image_mean
        out = self.head(x)
        speed_acuu = x.new_zeros(1)
        for module in self.body:
            if self.width_search:
                speed_curr = self.speed_estimator.estimateByMask(module, self.mask)
            else:
                speed_curr = self.speed_estimator.estimateByChannelNum(module)
            out = self.mask(out)
            out, speed_acuu = module(out, speed_curr, speed_acuu)

        if self.width_search:
            out = self.mask(out)
        out = self.tail(out)
        out = self.shuf(out)
        out = self.image_mean + out
        return out, speed_acuu

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @torch.no_grad()
    def get_current_blocks(self):
        num_blocks = 0
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                if module.alpha1 < module.alpha2:
                    num_blocks += 1
        return int(num_blocks)

    @torch.no_grad()
    def get_block_status(self):
        remain_block_idx = []
        for idx, module in enumerate(self.body.children()):
            if isinstance(module, AggregationLayer):
                alpha1, alpha2 = F.softmax(torch.stack([module.alpha1, module.alpha2]))
                if alpha1 < alpha2:
                    remain_block_idx.append(idx)
        return  remain_block_idx

    @torch.no_grad()
    def get_width_from_block_idx(self, remain_block_idx):
        @torch.no_grad()
        def _get_width_from_weight(w):
            return int(rounding(w).sum())

        all_width = []
        for idx, module in enumerate(self.body.children()):
            width = []
            # width_2 = []
            if idx in remain_block_idx and isinstance(module, AggregationLayer):
                width.append(_get_width_from_weight(self.mask.weight))
                for m in module.body_1.children():
                    if isinstance(m, BinaryConv2d):
                        width.append(_get_width_from_weight(self.mask.weight))
                # for n in module.body_2.children():
                #     if isinstance(n, BinaryConv2d):
                #         width.append(_get_width_from_weight(n.weight))
                all_width.append(width)
        return all_width

    @torch.no_grad()
    def get_alpha_grad(self):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                return module.alpha1.grad, module.alpha2.grad

    @torch.no_grad()
    def get_alpha(self):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                return module.alpha1, module.alpha2

    @torch.no_grad()
    def length_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                module.alpha1.requires_grad = flag
                module.alpha2.requires_grad = flag
                module.beta1.requires_grad = flag
                module.beta2.requires_grad = flag

    @torch.no_grad()
    def mask_grad(self, flag=False):
        for module in self.body.children():
            if isinstance(module, AggregationLayer):
                for m in module.body_1.children():
                    if isinstance(m, BinaryConv2d):
                        m.weight.requires_grad = flag
                for n in module.body_2.children():
                    if isinstance(n, BinaryConv2d):
                        n.weight.requires_grad = flag
        self.mask.weight.requires_grad = flag

    @torch.no_grad()
    def get_mask_grad(self):
        return self.mask.weight.grad

    @torch.no_grad()
    def get_mask_weight(self):
        return self.mask.weight.data

    @torch.no_grad()
    def load_pretrained(self):
        import os
        path, filename = os.path.split(__file__)
        weight_path = f'{path}/pretrained_weights'
        state_dict = torch.load(f'{weight_path}/repsr_b_x{self.scale}_{self.num_blocks}_{self.num_residual_units}.pt')
        state_dict_iterator = iter(state_dict.items())
        load_name, load_param = next(state_dict_iterator)
        for p in self.parameters():
            if p.size() == load_param.size():
                p.data = load_param
                try:
                    load_name, load_param = next(state_dict_iterator)
                except StopIteration:
                    pass



class Block(nn.Module):
    def __init__(self,
                 num_residual_units,
                 kernel_size,
                 weight_norm,
                 width_search=False):
        super(Block, self).__init__()
        body_1 = []
        body_2 = []
        multiplier = 2

        # 接下来对应 RepSR 的模块结构像其中添加 mask
        # 对第一个 3x3_bn_1x1 添加 mask
        conv_1 = weight_norm(
            nn.Conv2d(
                num_residual_units,
                num_residual_units * multiplier,
                kernel_size,
                padding= kernel_size // 2))

        body_1.append(conv_1)
        # RepSR block 当中，两个连续的卷积之间没有激活
        body_1.append(nn.BatchNorm2d(num_features=num_residual_units * multiplier))

        if width_search:
            body_1.append(BinaryConv2d(in_channels=int(num_residual_units * multiplier),
                                     out_channels=int(num_residual_units * multiplier),
                                     groups=int(num_residual_units * multiplier)))

        conv_1 = weight_norm(nn.Conv2d(
            num_residual_units * multiplier,
            num_residual_units,
            kernel_size = 1,
            padding= 1 // 2))

        body_1.append(conv_1)

        if width_search:
            body_1.append(BinaryConv2d(
                in_channels=num_residual_units,
                out_channels=num_residual_units,
                groups=num_residual_units))

        conv_2 = weight_norm(nn.Conv2d(
            num_residual_units,
            num_residual_units * multiplier,
            kernel_size=kernel_size,
            padding=kernel_size // 2))

        body_2.append(conv_2)
        body_2.append(nn.BatchNorm2d(num_residual_units * multiplier))

        if width_search:
            body_2.append(BinaryConv2d(
                in_channels=int(num_residual_units * multiplier),
                out_channels=int(num_residual_units * multiplier),
                groups=int(num_residual_units * multiplier)))

        conv_2 = weight_norm(nn.Conv2d(
            num_residual_units * multiplier,
            num_residual_units,
            kernel_size=1,
            padding=1 // 2))

        body_2.append(conv_2)

        if width_search:
            body_2.append(BinaryConv2d(
                in_channels=num_residual_units,
                out_channels=num_residual_units,
                groups=num_residual_units))

        self.body_1 = nn.Sequential(*body_1)
        self.body_2 = nn.Sequential(*body_2)

        self.init_weights()

    def forward(self, x):
        out = x + self.body_1(x) + self.body_2(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class AggregationLayer(Block):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

        # skip:
        self.alpha1 = nn.Parameter(torch.empty(1), requires_grad=True)
        self.beta1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        init.uniform_(self.alpha1, 0, 0.2)

        # preserve:
        self.alpha2 = nn.Parameter(torch.empty(1), requires_grad=True)
        self.beta2 = nn.Parameter(torch.ones(1), requires_grad=True)
        init.uniform_(self.alpha2, 0.8, 1)

    def forward(self, x, speed_accu, speed_curr):
        if self.training:
            # Get skip result
            sr1 = x
            # Get block result
            sr2 = x + self.body_1(x) + self.body_2(x)

            beta1, beta2 = ConditionFunction.apply(self.alpha1, self.alpha2, self.beta1, self.beta2)
            self.beta1.data, self.beta2.data = beta1, beta2

            x = beta1 * sr1 + beta2 * sr2

            speed_accu = beta2 * speed_curr + speed_accu

            return x, speed_accu
        else:
            if self.alpha1  >= self.alpha2:
                pass
            else:
                x = x + self.body_1(x) + self.body_2(x)
            speed_accu = self.beta2 * speed_curr + speed_accu
            return x, speed_accu

    def get_num_channels(self):
        channels = []
        for m in self.body_1.children():
            if isinstance(m, nn.Conv2d):
                channels.append(m.in_channels)
        channels.append(channels[0])
        return channels


class AggregationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sr1, sr2, speed_accu, speed_curr, alpha1, alpha2, beta1, beta2):
        ctx.num_outputs = 2
        '''这里的 ctx 是一个上下文对象，通常用于在前向传播和反向传播之间传递一些临时信息或缓存中间结果，
        在Pytorch中，ctx 对象会在创建 AggregationFunction 的实例时隐式的初始化，这里我们设定了 ctx
        的一个属性 num_outputs，用来存储一个名为num_outputs的值，以便在后续的计算中使用'''

        with torch.no_grad():
            if alpha1 > alpha2:
                beta1.data = beta1.new_ones(1)
                beta2.data = beta2.new_zeros(1)
            else:
                beta1.data = beta1.new_zeros(1)
                beta2.data = beta2.new_ones(1)

            ctx.save_for_backward(sr1, sr2, speed_accu, speed_curr, beta1, beta2)

            sr = sr1 * beta1 + sr2 * beta2
            total_speed = beta2 * speed_curr + speed_accu

            return  sr, total_speed

    @staticmethod
    def backward(ctx, grad_output_sr, grad_output_speed):
        sr1, sr2, speed_accu, speed_curr, beta1, beta2 = ctx.saved_tensors
        grad_sr1 = grad_output_sr * beta1
        grad_sr2 = grad_output_sr * beta2
        grad_speed_accu = grad_output_speed * beta2
        grad_speed_curr = grad_output_speed
        grad_beta1 = grad_output_sr.bmm(beta1)   # for grad_alpha1
        grad_beta2 = grad_output_sr * beta2 + grad_output_speed * speed_curr  # for grad_alpha2

        grad_alpha1 = grad_beta1
        grad_alpha2 = grad_beta2

        return grad_sr1, grad_sr2, grad_speed_accu, grad_speed_curr, grad_alpha1, grad_alpha2, None, None

class ConditionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha1, alpha2, beta1, beta2):
        with torch.no_grad():
            if alpha1 >= alpha2:
                beta1.data = beta1.new_ones(1)
                beta2.data = beta2.new_zeros(1)
            else:
                beta1.data = beta1.new_zeros(1)
                beta2.data = beta2.new_ones(1)

        return beta1, beta2

    @staticmethod
    def backward(ctx, grad_output_beta1, grad_output_beta2):

        grad_alpha1 = grad_output_beta1
        grad_alpha2 = grad_output_beta2

        return grad_alpha1, grad_alpha2, None, None


if __name__ == '__main__':
    sr1 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    sr2 = torch.rand([20, 30], requires_grad=True, dtype=torch.float64)
    speed_accu = torch.rand(1, requires_grad=True, dtype=torch.float64)
    speed_curr = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    a2 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b1 = torch.rand(1, requires_grad=True, dtype=torch.float64)
    b2 = torch.rand(1, requires_grad=True, dtype=torch.float64)

    torch.autograd.gradcheck(ConditionFunction.apply, (a1, a2, b1, b2),
                             eps=1e-1)
    pass

