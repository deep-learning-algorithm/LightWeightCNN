# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午3:43
@file: test_basic_conv2d.py
@author: zj
@description: 
"""

import torch
from models.basic_conv2d import BasicConv2d


def test():
    x = torch.randn((1, 3, 28, 28))
    model = BasicConv2d(3, 10, kernel_size=3, padding=1)
    outputs = model(x)
    assert len(outputs.shape) == 4
