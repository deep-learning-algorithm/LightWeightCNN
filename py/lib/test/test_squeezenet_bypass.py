# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午7:53
@file: test_squeezenet_bypass.py
@author: zj
@description: 
"""

import torch
from models.squeeze_net_bypass import SqueezeNetBypass


def test():
    num_classes = 20

    x = torch.randn((1, 3, 224, 224))
    model = SqueezeNetBypass(num_classes=num_classes)
    outputs = model(x)

    assert len(outputs.shape) == 4
    assert outputs.shape[1] == num_classes
