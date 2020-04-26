# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午3:35
@file: util.py
@author: zj
@description: 
"""

import torch
import os


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
