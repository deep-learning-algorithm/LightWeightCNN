# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 上午9:07
@file: conv_bn_relu.py
@author: zj
@description: 
"""

import torch.nn as nn


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
