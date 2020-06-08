# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 下午2:21
@file: channel_shuffle.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()

        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
