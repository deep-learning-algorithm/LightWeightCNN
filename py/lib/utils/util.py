# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午3:35
@file: util.py
@author: zj
@description: 
"""

import torch
import os
import matplotlib.pyplot as plt


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def save_model(model, model_save_path):
    # 保存最好的模型参数
    check_dir('./models')
    torch.save(model.state_dict(), model_save_path)


def save_png(title, res_dict):
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    fig = plt.figure()

    plt.title(title)
    for name, res in res_dict.items():
        for k, v in res.items():
            x = list(range(len(v)))
            plt.plot(v, label='%s-%s' % (name, k))

    plt.legend()
    plt.savefig('%s.png' % title)
