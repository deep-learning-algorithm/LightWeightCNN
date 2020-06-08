# -*- coding: utf-8 -*-

"""
@date: 2020/6/8 下午4:19
@file: test_mobilenet_v2.py
@author: zj
@description: 
"""

import torch
from models.mobilenet_v2.mobilenet_v2 import MobileNetV2


def test():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = MobileNetV2(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)
