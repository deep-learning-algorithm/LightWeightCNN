# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午4:55
@file: squeeze_net_bypass.py
@author: zj
@description: 
"""

import torch.nn as nn

from models.squeezenet.basic_conv2d import BasicConv2d
from models.squeezenet.fire import Fire
from models.squeezenet.fire_bypass import FireBypass


class SqueezeNetBypass(nn.Module):

    def __init__(self, num_classes=1000):
        super(SqueezeNetBypass, self).__init__()
        conv_block = BasicConv2d
        fire_block = Fire
        fire_bypass_block = FireBypass

        self.conv1 = conv_block(3, 96, kernel_size=7, stride=2, padding=2)
        self.max_pool = nn.MaxPool2d(3, stride=2)

        self.fire2 = fire_block(96, 16, 64, 64)
        self.fire_bypass3 = fire_bypass_block(128, 16, 64, 64)
        self.fire4 = fire_block(128, 32, 128, 128)
        self.fire_bypass5 = fire_bypass_block(256, 32, 128, 128)
        self.fire6 = fire_block(256, 48, 192, 192)
        self.fire_bypass7 = fire_bypass_block(384, 48, 192, 192)
        self.fire8 = fire_block(384, 64, 256, 256)
        self.fire_bypass9 = fire_bypass_block(512, 64, 256, 256)
        self.conv10 = conv_block(512, num_classes, kernel_size=1)

        self.avg_pool = nn.AvgPool2d(13, stride=1)

    def forward(self, x):
        assert len(x.shape) == 4
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 96 x 111 x 111
        x = self.max_pool(x)
        # N x 96 x 55 x 55
        x = self.fire2(x)
        # N x 128 x 55 x 55
        x = self.fire_bypass3(x)
        # N x 128 x 55 x 55
        x = self.fire4(x)
        # N x 256 x 55 x 55
        x = self.max_pool(x)
        # N x 256 x 27 x 27
        x = self.fire_bypass5(x)
        # N x 256 x 27 x 27
        x = self.fire6(x)
        # N x 384 x 27 x 27
        x = self.fire_bypass7(x)
        # N x 384 x 27 x 27
        x = self.fire8(x)
        # N x 512 x 27 x 27
        x = self.max_pool(x)
        # N x 512 x 13 x 13
        x = self.fire_bypass9(x)
        # N x 512 x 13 x 13
        x = self.conv10(x)
        # N x C x 13 x 13
        x = self.avg_pool(x)
        # N x C x 1 x 1
        return x.squeeze()