# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 下午2:06
@file: shufflenet_unit.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_shuffle import ChannelShuffle


class ShuffleNetUnit(nn.Module):

    def __init__(self, inp, oup, stride, groups=1):
        super(ShuffleNetUnit, self).__init__()
        assert stride in [1, 2]

        self.stride = stride

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
            dw_out_channels = inp * (oup // inp)
            gconv_oup = oup - inp
        else:
            self.branch1 = nn.Sequential()
            dw_out_channels = oup
            gconv_oup = oup

        self.branch2 = nn.Sequential(
            # 分组卷积
            nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            ChannelShuffle(groups),
            # 深度卷积
            nn.Conv2d(inp, dw_out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=inp),
            nn.BatchNorm2d(dw_out_channels),
            # 分组卷积
            nn.Conv2d(dw_out_channels, gconv_oup, kernel_size=1, stride=1, padding=0, bias=False, groups=groups),
            nn.BatchNorm2d(gconv_oup),
            nn.ReLU(inplace=True),
        )

    def _forward_impl(self, x):
        if self.stride == 1:
            out = x + self.branch2(x)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = F.relu(out)

        return out

    def forward(self, x):
        return self._forward_impl(x)
