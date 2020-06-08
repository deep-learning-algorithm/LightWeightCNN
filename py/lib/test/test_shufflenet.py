# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 下午2:37
@file: test_shufflenet.py
@author: zj
@description: 
"""

import torch
from models.shufflenet.shufflenet import shufflenet_g_1
from models.shufflenet.shufflenet import shufflenet_g_2
from models.shufflenet.shufflenet import shufflenet_g_3
from models.shufflenet.shufflenet import shufflenet_g_4
from models.shufflenet.shufflenet import shufflenet_g_8


def test_g_1():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = shufflenet_g_1(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)


def test_g_2():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = shufflenet_g_2(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)


def test_g_3():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = shufflenet_g_3(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)


def test_g_4():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = shufflenet_g_4(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)


def test_g_8():
    N = 8
    num_classes = 20

    x = torch.randn((N, 3, 224, 224))
    model = shufflenet_g_8(num_classes=num_classes)
    outputs = model(x)
    assert outputs.shape == (N, num_classes)
