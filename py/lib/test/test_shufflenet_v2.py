# -*- coding: utf-8 -*-

"""
@date: 2020/6/8 下午4:25
@file: test_shufflenet_v2.py
@author: zj
@description: 
"""

import torch
from models.shufflenet_v2.shufflenet_v2 import shufflenet_v2_x0_5
from models.shufflenet_v2.shufflenet_v2 import shufflenet_v2_x1_0
from models.shufflenet_v2.shufflenet_v2 import shufflenet_v2_x1_5
from models.shufflenet_v2.shufflenet_v2 import shufflenet_v2_x2_0


class TestShuffleNetV2(object):

    def test_shufflenet_v2_x0_5(self):
        N = 8
        num_classes = 20

        x = torch.randn((N, 3, 224, 224))
        model = shufflenet_v2_x0_5(num_classes=num_classes)
        outputs = model(x)
        assert outputs.shape == (N, num_classes)

    def test_shufflenet_v2_x1_0(self):
        N = 8
        num_classes = 20

        x = torch.randn((N, 3, 224, 224))
        model = shufflenet_v2_x1_0(num_classes=num_classes)
        outputs = model(x)
        assert outputs.shape == (N, num_classes)

    def test_shufflenet_v2_x1_5(self):
        N = 8
        num_classes = 20

        x = torch.randn((N, 3, 224, 224))
        model = shufflenet_v2_x1_5(num_classes=num_classes)
        outputs = model(x)
        assert outputs.shape == (N, num_classes)

    def test_shufflenet_v2_x2_0(self):
        N = 8
        num_classes = 20

        x = torch.randn((N, 3, 224, 224))
        model = shufflenet_v2_x2_0(num_classes=num_classes)
        outputs = model(x)
        assert outputs.shape == (N, num_classes)
