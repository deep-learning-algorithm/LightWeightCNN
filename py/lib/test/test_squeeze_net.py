# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午4:07
@file: test_squeeze_net.py
@author: zj
@description: 
"""

import torch
from models.squeeze_net import SqueezeNet


def test():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = SqueezeNet(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)

