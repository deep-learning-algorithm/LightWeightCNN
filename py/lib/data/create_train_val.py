# -*- coding: utf-8 -*-

"""
@date: 2020/3/26 下午2:50
@file: create_train_val.py
@author: zj
@description: 提取分类任务的训练/测试集，分类别保存
"""

import cv2
import numpy as np
import os
import xmltodict

#### for train
# aeroplane 1171
# bicycle 1064
# bird 1605
# boat 1140
# bottle 1764
# bus 822
# car 3267
# cat 1593
# chair 3152
# cow 847
# diningtable 824
# dog 2025
# horse 1072
# motorbike 1052
# person 13256
# pottedplant 1487
# sheep 1070
# sofa 814
# train 925
# tvmonitor 1108
# total train num: 40058
#### for test
# aeroplane 285
# bicycle 337
# bird 459
# boat 263
# bottle 469
# bus 213
# car 1201
# cat 358
# chair 756
# cow 244
# diningtable 206
# dog 489
# horse 348
# motorbike 325
# person 4528
# pottedplant 480
# sheep 242
# sofa 239
# train 282
# tvmonitor 308
# total test num: 12032

alphabets = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def find_all_cate_rects(annotation_dir, name_list):
    """
    找出所有的类别的标注框（取消标记为difficult的边界框）
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
    # 前提条件：足够的内存!!!
    # image_dict = dict()
    # image_name_list = os.listdir(image_dir)
    # for name in image_name_list:
    #     image_path = os.path.join(image_dir, name)
    #     img = cv2.imread(image_path)
    #
    #     image_dict[name.split('.')[0]] = img

    # 遍历所有类别，保存标注的图像
    for i in range(20):
        cate_name = alphabets[i]
        cate_dir = os.path.join(res_dir, cate_name)
        check_dir(cate_dir)

        for item in cate_list[i]:
            img_name = item['img_name']
            xmin, ymin, xmax, ymax = item['rect']

            image_path = os.path.join(image_dir, img_name+'.jpg')
            img = cv2.imread(image_path)
            rect_img = img[ymin:ymax, xmin:xmax]
            # rect_img = image_dict[img_name][ymin:ymax, xmin:xmax]
            img_path = os.path.join(cate_dir, '%s-%d-%d-%d-%d.png' % (img_name, xmin, ymin, xmax, ymax))
            cv2.imwrite(img_path, rect_img)


if __name__ == '__main__':
    root_dir = '../../data/pascal-voc/'
    train_txt_path = '../../data/pascal-voc/train/name.csv'
    val_txt_path = '../../data/pascal-voc/test/name.csv'

    for phase in ['train', 'test']:
        if phase == 'train':
            suffix = 'train_imgs'
        else:
            suffix = 'test_imgs'
        dst_dir = os.path.join(root_dir, suffix)
        check_dir(dst_dir)
        print(dst_dir)

        name_path = os.path.join(root_dir, phase, 'name.csv')
        name_list = np.loadtxt(name_path, dtype=np.str, delimiter=' ')

        annotation_dir = os.path.join(root_dir, phase, 'Annotations')
        rects_list = find_all_cate_rects(annotation_dir, name_list)

        total_num = 0
        # 打印出每个类别的数据
        for i in range(20):
            total_num += len(rects_list[i])
            print(alphabets[i], len(rects_list[i]))
        print('total {} num: {}'.format(phase, total_num))

        image_dir = os.path.join(root_dir, phase, 'JPEGImages')
        save_cate(rects_list, image_dir, dst_dir)

        print()
    print('done')
