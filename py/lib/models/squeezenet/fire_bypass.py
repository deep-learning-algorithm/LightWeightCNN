# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午4:40
@file: fire_bypass.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn
from models.squeezenet.basic_conv2d import BasicConv2d


class FireBypass(nn.Module):
    """
    两层网络，卷积核大小固定为3x3，每一层的滤波器个数相同
    """

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, norm_layer=None):
        super(FireBypass, self).__init__()
        conv_block = BasicConv2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = inplanes
        self.squeeze = conv_block(inplanes, squeeze_planes)
        self.expand1x1 = self.conv1x1(squeeze_planes, expand1x1_planes)
        self.bn1 = norm_layer(expand1x1_planes)
        self.expand3x3 = self.conv3x3(squeeze_planes, expand3x3_planes)
        self.bn2 = norm_layer(expand3x3_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.squeeze(x)
        out_1 = self.expand1x1(out)
        out_1 = self.bn1(out_1)
        out_2 = self.expand3x3(out)
        out_2 = self.bn2(out_2)
        out = torch.cat([out_1, out_2], 1)

        out += identity
        out = self.relu(out)

        return out

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)
