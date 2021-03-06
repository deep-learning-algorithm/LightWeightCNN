# LightWeightCNN

[![Documentation Status](https://readthedocs.org/projects/lightweightcnn/badge/?version=latest)](https://lightweightcnn.readthedocs.io/zh_CN/latest/?badge=latest) [![standard-readme compliant](https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme) [![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org) [![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

>轻量化卷积神经网络实现

| CNN Architecture | Data Type (bit) | Model Size (MB) | GFlops （1080Ti） | Top-1 Acc(VOC 07+12) | Top-5 Acc(VOC 07+12) |
|:----------------:|:---------------:|:---------------:|:-----------------:|:--------------------:|:--------------------:|
|      AlexNet     |        32       |     233.081     |       1.429       |        68.24%        |        94.22%        |
|    SqueezeNet    |        32       |      4.793      |       1.692       |        75.46%        |        96.78%        |
| SqueezeNetBypass |        32       |      4.793      |       1.692       |        77.54%        |        97.41%        |

## 内容列表

- [LightWeightCNN](#lightweightcnn)
  - [内容列表](#内容列表)
  - [背景](#背景)
  - [安装](#安装)
    - [文档工具安装](#文档工具安装)
    - [Python库依赖](#python库依赖)
  - [用法](#用法)
    - [文档浏览](#文档浏览)
  - [主要维护人员](#主要维护人员)
  - [致谢](#致谢)
    - [引用](#引用)
  - [参与贡献方式](#参与贡献方式)
  - [许可证](#许可证)

## 背景

* [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile](https://arxiv.org/abs/1707.01083)
* [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
 
## 安装

### 文档工具安装

```
$ pip install -r requirements.txt
```

### Python库依赖

```
$ cd py
$ pip install -r requirements.txt
```

## 用法

### 文档浏览

有两种使用方式

1. 在线浏览文档：[LightWeightCNN](https://lightweightcnn.readthedocs.io/zh_CN/latest/)

2. 本地浏览文档，实现如下：

    ```
    $ git clone https://github.com/zjZSTU/LightWeightCNN.git
    $ cd LightWeightCNN
    $ mkdocs serve
    ```
    启动本地服务器后即可登录浏览器`localhost:8000`

## 主要维护人员

* zhujian - *Initial work* - [zjZSTU](https://github.com/zjZSTU)

## 致谢

### 引用

```
@misc{i2016squeezenet,
    title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size},
    author={Forrest N. Iandola and Song Han and Matthew W. Moskewicz and Khalid Ashraf and William J. Dally and Kurt Keutzer},
    year={2016},
    eprint={1602.07360},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{howard2017mobilenets,
    title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
    author={Andrew G. Howard and Menglong Zhu and Bo Chen and Dmitry Kalenichenko and Weijun Wang and Tobias Weyand and Marco Andreetto and Hartwig Adam},
    year={2017},
    eprint={1704.04861},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{s2018mobilenetv2,
    title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
    author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
    year={2018},
    eprint={1801.04381},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{zhang2017shufflenet,
    title={ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices},
    author={Xiangyu Zhang and Xinyu Zhou and Mengxiao Lin and Jian Sun},
    year={2017},
    eprint={1707.01083},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{ma2018shufflenet,
    title={ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design},
    author={Ningning Ma and Xiangyu Zhang and Hai-Tao Zheng and Jian Sun},
    year={2018},
    eprint={1807.11164},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}

@misc{pascal-voc-2012,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"}
```

## 参与贡献方式

欢迎任何人的参与！打开[issue](https://github.com/zjZSTU/LightWeightCNN/issues)或提交合并请求。

注意:

* `GIT`提交，请遵守[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)规范
* 语义版本化，请遵守[Semantic Versioning 2.0.0](https://semver.org)规范
* `README`编写，请遵守[standard-readme](https://github.com/RichardLitt/standard-readme)规范

## 许可证

[Apache License 2.0](LICENSE) © 2020 zjZSTU
