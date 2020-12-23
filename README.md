# UDA

Simple implementation of [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)(NIPS 2020)

## Overview

## Dataset
- CIFAR10

## Result
|   Model   | Number of labeled examples | RandAugment | Top-1 Accuracy |
| :-------: | :------------------------: | :---------: | :------------: |
| ResNet-50 |           40000            |      X      |       -        |
| ResNet-50 |           40000            |      O      |       -        |
| ResNet-50 |            4000            |      O      |       -        |
|    UDA    |            4000            |      O      |       -        |

## Requirements
```sh
# for Windows
>>> pip install -r requirements.txt
```

## Reference
- [[Paper] Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)
- [ResNet Code from lepoeme20](https://github.com/lepoeme20/pytorch-image-classification)
- [RandAugment Code from ildoonet](https://github.com/ildoonet/pytorch-randaugment)
