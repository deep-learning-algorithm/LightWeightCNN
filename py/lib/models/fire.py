# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午3:39
@file: fire.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from models.basic_conv2d import BasicConv2d


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        conv_block = BasicConv2d

        self.inplanes = inplanes
        self.squeeze = conv_block(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = conv_block(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = conv_block(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)
