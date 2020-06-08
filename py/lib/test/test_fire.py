# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午3:49
@file: test_fire.py
@author: zj
@description: 
"""

import torch
from models.squeezenet.fire import Fire


def test():
    x = torch.randn((1, 3, 28, 28))
    model = Fire(3, 10, 5, 5)
    outputs = model(x)
    assert len(outputs.shape) == 4
