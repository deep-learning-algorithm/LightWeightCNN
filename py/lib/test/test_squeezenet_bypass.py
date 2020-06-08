# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午7:53
@file: test_squeezenet_bypass.py
@author: zj
@description: 
"""

import torch
from models.squeezenet.squeeze_net_bypass import SqueezeNetBypass


def test():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = SqueezeNetBypass(num_classes=num_classes)
    outputs = model(x)

    assert outputs.shape == (N, num_classes)
