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
    num_classes = 20

    x = torch.randn((1, 3, 224, 224))
    model = SqueezeNet(num_classes=num_classes)
    outputs = model(x)
    assert len(outputs.shape) == 4
    assert outputs.shape[1] == num_classes
