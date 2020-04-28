# -*- coding: utf-8 -*-

"""
@date: 2020/4/27 下午8:25
@file: metrics.py
@author: zj
@description: 
"""

import torch
from thop import profile

from torchvision.models import AlexNet
from models.squeeze_net import SqueezeNet
from models.squeeze_net_bypass import SqueezeNetBypass


def compute_num_flops(model):
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    # print(macs, params)

    GFlops = macs * 2.0 / pow(10, 9)
    params_size = params * 4.0 / 1024 / 1024
    return GFlops, params_size


def topk_accuracy(output, target, topk=(1,)):
    """
    计算前K个。N表示样本数，C表示类别数
    :param output: 大小为[N, C]，每行表示该样本计算得到的C个类别概率
    :param target: 大小为[N]，每行表示指定类别
    :param topk: tuple，计算前top-k的accuracy
    :return: list
    """
    assert len(output.shape) == 2 and output.shape[0] == target.shape[0]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    for name in ['alexnet', 'squeezenet', 'squeezenet-bypass']:
        if name == 'alexnet':
            model = AlexNet()
        elif name == 'squeezenet':
            model = SqueezeNet()
        else:
            model = SqueezeNetBypass()
        gflops, params_size = compute_num_flops(model)
        print('{}: {:.3f} GFlops - {:.3f} MB'.format(name, gflops, params_size))