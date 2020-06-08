# -*- coding: utf-8 -*-

"""
@date: 2020/6/7 上午9:15
@file: mobilenet.py
@author: zj
@description: 
"""

import torch.nn as nn

from .conv_bn_relu import ConvBNReLU
from .depthwise_separable_conv import DepthwiseSeparableConv


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        depth_sep_conv_setting = [
            # inp, oup, stride
            [32, 64, 1],
            [64, 128, 2],
            [128, 128, 1],
            [128, 256, 2],
            [256, 256, 1],
            [256, 512, 2],

            # repeat 5 times
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],
            [512, 512, 1],

            [512, 1024, 2],
            [1024, 1024, 2]
        ]

        features = [ConvBNReLU(3, 32, kernel_size=3, stride=2)]
        for inp, oup, stride in depth_sep_conv_setting:
            features.append(DepthwiseSeparableConv(inp, oup, stride))
        self.features = nn.Sequential(*features)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        self.init_param()

    def init_param(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


if __name__ == '__main__':
    import torch

    data = torch.randn(1, 3, 224, 224)
    model = MobileNet()
    outputs = model(data)

    print(outputs.shape)
