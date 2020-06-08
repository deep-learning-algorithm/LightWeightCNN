# -*- coding: utf-8 -*-

"""
@date: 2020/6/6 下午2:49
@file: depthwise_separable_conv.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from .conv_bn_relu import ConvBNReLU


class DepthwiseSeparableConv(nn.Sequential):

    def __init__(self, inp, oup, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        padding = 1 if stride == 2 else 0

        layers = []
        layers.extend([
            # depthwise
            ConvBNReLU(inp, oup, kernel_size=3, stride=stride, padding=padding, groups=inp),
            # pointwise
            ConvBNReLU(oup, oup, kernel_size=1, stride=1)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
