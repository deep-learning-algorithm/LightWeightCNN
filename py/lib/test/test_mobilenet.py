# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 上午9:32
@file: test_mobilenet.py
@author: zj
@description: 
"""

import torch
from models.mobilenet.mobilenet import MobileNet


def test():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = MobileNet(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)
