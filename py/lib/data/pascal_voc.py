# -*- coding: utf-8 -*-

"""
@date: 2020/4/6 下午4:14
@file: pascal_voc.py
@author: zj
@description: 解析07+12数据集
"""

import os
import cv2
import shutil
import numpy as np
from torchvision.datasets import VOCDetection

# 2007 trainval

trainval_07_annotations = '../../data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
trainval_07_image = '../../data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
trainval_07_txt = '../../data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'

# 2007 test

test_07_annotations = '../../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
test_07_image = '../../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages'
test_07_txt = '../../data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# 2012 trainval

trainval_12_annotations = '../../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'
trainval_12_image = '../../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'
trainval_12_txt = '../../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def parse_data(txt_path, annotation_dir, image_dir):
    """
    解析txt文件，返回相应的图像和标注文件
    :return:
    """
    name_list = np.loadtxt(txt_path, dtype=np.str, delimiter=' ')
    print(name_list)

    annotation_list = [os.path.join(annotation_dir, name + ".xml") for name in name_list]
    image_list = [os.path.join(image_dir, name + ".jpg") for name in name_list]

    return name_list, annotation_list, image_list


if __name__ == '__main__':
    data_dir = '../../data/pascal-voc'
    check_dir(data_dir)

    txt_list = [trainval_07_txt, trainval_12_txt, test_07_txt]
    annotation_list = [trainval_07_annotations, trainval_12_annotations, test_07_annotations]
    image_list = [trainval_07_image, trainval_12_image, test_07_image]

    total_train_list = list()
    total_test_list = list()

    for txt_path, annotation_dir, image_dir in zip(txt_list, annotation_list, image_list):
        print(txt_path, annotation_dir, image_dir)
        name_list, annotation_list, image_list = parse_data(txt_path, annotation_dir, image_dir)

        if 'trainval' in txt_path:
            suffix = 'train'
            total_train_list.extend(name_list)
        else:
            suffix = 'test'
            total_test_list.extend(name_list)

        # 新建结果文件夹
        dst_dir = os.path.join(data_dir, suffix)
        check_dir(dst_dir)
        dst_annotation_dir = os.path.join(dst_dir, 'Annotations')
        check_dir(dst_annotation_dir)
        dst_image_dir = os.path.join(dst_dir, 'JPEGImages')
        check_dir(dst_image_dir)

        # 依次复制标注文件和图像
        for name, src_annotation_path, src_image_path in zip(name_list, annotation_list, image_list):
            dst_annotation_path = os.path.join(dst_annotation_dir, name + ".xml")
            dst_image_path = os.path.join(dst_image_dir, name + ".jpg")

            shutil.copyfile(src_annotation_path, dst_annotation_path)
            shutil.copyfile(src_image_path, dst_image_path)

    print('train num: {}, test num: {}'.format(len(total_train_list), len(total_test_list)))

    # 保存文件名
    train_dir = os.path.join(data_dir, 'train', 'name.csv')
    np.savetxt(train_dir, total_train_list, fmt='%s', delimiter=' ')
    test_dir = os.path.join(data_dir, 'test', 'name.csv')
    np.savetxt(test_dir, total_test_list, fmt='%s', delimiter=' ')

    print('done')
