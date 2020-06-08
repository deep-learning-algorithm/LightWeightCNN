# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 上午9:54
@file: shufflenet.py
@author: zj
@description: 
"""

import torch.nn as nn
from torchvision.models import shufflenetv2
from .shufflenet_unit import ShuffleNetUnit


class ShuffleNet(nn.Module):

    def __init__(self, stages_repeats, stages_out_channels, groups=1, num_classes=1000):
        super(ShuffleNet, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 4:
            raise ValueError('expected stages_out_channels as list of 4 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleNetUnit(input_channels, output_channels, 2, groups=groups)]
            for i in range(repeats - 1):
                seq.append(ShuffleNetUnit(output_channels, output_channels, 1, groups=groups))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def shufflenet_g_1(num_classes=1000):
    return ShuffleNet([4, 8, 4], [24, 144, 288, 576], num_classes=num_classes)


def shufflenet_g_2(num_classes=1000):
    return ShuffleNet([4, 8, 4], [24, 200, 400, 800], num_classes=num_classes)


def shufflenet_g_3(num_classes=1000):
    return ShuffleNet([4, 8, 4], [24, 240, 480, 960], num_classes=num_classes)


def shufflenet_g_4(num_classes=1000):
    return ShuffleNet([4, 8, 4], [24, 272, 544, 1088], num_classes=num_classes)


def shufflenet_g_8(num_classes=1000):
    return ShuffleNet([4, 8, 4], [24, 384, 768, 1536], num_classes=num_classes)
