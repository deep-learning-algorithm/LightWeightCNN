
# 数据集

参考：[[数据集][PASCAL VOC]07+12](https://blog.zhujian.life/posts/db93f7d2.html)

本仓库使用`PASCAL VOC 07 + 12`作为分类数据集

## 简介

`PASCAL VOC`数据集包含了`20`类

* `Person: person`
* `Animal: bird, cat, cow（奶牛）, dog, horse, sheep（绵羊）`
* `Vehicle（交通工具）: aeroplane（飞机）, bicycle, boat（小船）, bus（公共汽车）, car（轿车）, motorbike（摩托车）, train（火车）`
* `Indoor（室内）: bottle（瓶子）, chair（椅子）, dining table（餐桌）, potted plant（盆栽植物）, sofa, tv/monitor（电视/显示器）`

## 07+12

`07 + 12`数据集合并后，共得到如下数据：

1. 训练数据：`16551`张图像，共`40058`个目标
2. 测试素据：`4952`张图像，共`12032`个目标

## 实现

* `py/lib/data/pascal_voc.py`
* `py/lib/data/create_train_val.py`