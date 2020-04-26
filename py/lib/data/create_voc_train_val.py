# -*- coding: utf-8 -*-

"""
@date: 2020/3/26 下午2:50
@file: create_voc_train_val.py
@author: zj
@description: 提取全部的训练/验证集，分类别保存标注信息
"""

import cv2
import numpy as np
import os
import xmltodict

import utils.util as util

root_dir = '../../data/train_val/'
train_dir = '../../data/train_val/train/'
val_dir = '../../data/train_val/val/'

train_txt_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val_txt_path = '../../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt'

annotation_dir = '../../data/VOCdevkit/VOC2007/Annotations'
jpeg_image_dir = '../../data/VOCdevkit/VOC2007/JPEGImages'

alphabets = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def find_all_cate_rects(annotation_dir, name_list):
    """
    找出所有的类别的标注框
    """
    cate_list = list()
    for i in range(20):
        cate_list.append(list())

    for name in name_list:
        annotation_path = os.path.join(annotation_dir, name + ".xml")
        with open(annotation_path, 'rb') as f:
            xml_dict = xmltodict.parse(f)
            # print(xml_dict)

            objects = xml_dict['annotation']['object']
            if isinstance(objects, list):
                for obj in objects:
                    obj_name = obj['name']
                    obj_idx = alphabets.index(obj_name)

                    difficult = int(obj['difficult'])
                    if difficult != 1:
                        bndbox = obj['bndbox']
                        cate_list[obj_idx].append({'img_name': name, 'rect':
                            (int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax']))})
            elif isinstance(objects, dict):
                obj_name = objects['name']
                obj_idx = alphabets.index(obj_name)

                difficult = int(objects['difficult'])
                if difficult != 1:
                    bndbox = objects['bndbox']
                    cate_list[obj_idx].append({'img_name': name, 'rect':
                        (int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax']))})
            else:
                pass

    return cate_list


def save_cate(cate_list, image_dir, res_dir):
    """
    保存裁剪的图像
    """
    # 保存image_dir下所有图像，以便后续查询
    image_dict = dict()
    image_name_list = os.listdir(image_dir)
    for name in image_name_list:
        image_path = os.path.join(image_dir, name)
        img = cv2.imread(image_path)

        image_dict[name.split('.')[0]] = img

    # 遍历所有类别，保存标注的图像
    for i in range(20):
        cate_name = alphabets[i]
        cate_dir = os.path.join(res_dir, cate_name)
        util.check_dir(cate_dir)

        for item in cate_list[i]:
            img_name = item['img_name']
            xmin, ymin, xmax, ymax = item['rect']

            rect_img = image_dict[img_name][ymin:ymax, xmin:xmax]
            img_path = os.path.join(cate_dir, '%s-%d-%d-%d-%d.png' % (img_name, xmin, ymin, xmax, ymax))
            cv2.imwrite(img_path, rect_img)


if __name__ == '__main__':
    util.check_dir(root_dir)
    util.check_dir(train_dir)
    util.check_dir(val_dir)

    train_name_list = np.loadtxt(train_txt_path, dtype=np.str)
    print(train_name_list)
    cate_list = find_all_cate_rects(annotation_dir, train_name_list)
    print([len(x) for x in cate_list])
    save_cate(cate_list, jpeg_image_dir, train_dir)

    val_name_list = np.loadtxt(val_txt_path, dtype=np.str)
    print(val_name_list)
    cate_list = find_all_cate_rects(annotation_dir, val_name_list)
    print([len(x) for x in cate_list])
    save_cate(cate_list, jpeg_image_dir, val_dir)

    print('done')
