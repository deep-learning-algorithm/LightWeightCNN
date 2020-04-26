# -*- coding: utf-8 -*-

"""
@date: 2020/4/26 下午7:58
@file: train.py
@author: zj
@description: 
"""

import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import utils.util as util
from models.squeeze_net import SqueezeNet
from models.squeeze_net_bypass import SqueezeNetBypass


def load_data(data_root_dir):
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'test']:
        data_dir = os.path.join(data_root_dir, name)
        # print(data_dir)

        data_set = ImageFolder(data_dir, transform=transform)
        data_loader = DataLoader(data_set, batch_size=128, shuffle=True, num_workers=8)
        data_loaders[name] = data_loader
        data_sizes[name] = len(data_set)
    return data_loaders, data_sizes


def train_model(data_loaders, data_sizes, model_name, model, criterion, optimizer, lr_scheduler,
                num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    loss_dict = {'train': [], 'test': []}
    acc_dict = {'train': [], 'test': []}
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]
            loss_dict[phase].append(epoch_loss)
            acc_dict[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        # 每训练一轮就保存
        # util.save_model(model.cpu(), '../data/models/%s_%d.pth' % (model_name, epoch))
        # model = model.to(device)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model, loss_dict, acc_dict


if __name__ == '__main__':
    device = util.get_device()
    # device = 'cpu'

    data_loaders, data_sizes = load_data('../data/train_val')
    print(data_loaders)
    print(data_sizes)

    res_loss = dict()
    res_acc = dict()
    num_classes = 20
    for name in ['AlexNet', 'SqueezeNet', 'SqueezeNetBypass']:
        if name == 'SqueezeNet-18':
            model = SqueezeNet(num_classes=num_classes)
        elif name == 'SqueezeNetBypass':
            model = SqueezeNetBypass(num_classes=num_classes)
        else:
            model = alexnet(num_classes=num_classes)
        model.eval()
        # print(model)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.95)

        util.check_dir('../data/models/')
        best_model, loss_dict, acc_dict = train_model(data_loaders, data_sizes, name, model,
                                                      criterion, optimizer, lr_schduler, num_epochs=50, device=device)
        # 保存最好的模型参数
        # util.save_model(best_model.cpu(), '../data/models/best_%s.pth' % name)

        res_loss[name] = loss_dict
        res_acc[name] = acc_dict

        print('train %s done' % name)
        print()

    util.save_png('loss', res_loss)
    util.save_png('acc', res_acc)
