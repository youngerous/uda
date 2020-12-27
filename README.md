# UDA

Simple pytorch implementation of [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)(NIPS 2020).

## Structure
```sh
.gitignore
LICENSE
README.md
requirements.txt
src/
    ├── config.py
    ├── dataset.py
    ├── main.py
    ├── net.py
    ├── trainer.py
    ├── utils.py
    └── data/ # ignored in this repo
        └── cifar-10-batches-py/
            ├── batches.meta
            ├── data_batch_1
            ├── data_batch_2
            ├── data_batch_3
            ├── data_batch_4
            ├── data_batch_5
            ├── readme.html
            └── test_batch
```

## Dataset
- CIFAR-10 ([download link](https://www.cs.toronto.edu/~kriz/cifar.html))


## Requirements
```sh
# Windows
>>> pip install -r requirements.txt
```

## Run
**Train**
```sh
# train fully supervised model
>>> python main.py --lr 0.001 --train

# train fully supervised model with fewer samples
>>> python main.py --lr 0.001 --num_labeled {NUMBER_OF_SAMPLES} --train

# apply uda
>>> python main.py --num_labeled {NUMBER_OF_SAMPLES} --train --do_uda
```
**Test**
```sh
>>> python main.py --test
```

## Result

|     Model      | Number of labeled examples | Augmentation | Top-1 Accuracy |
| :------------: | :------------------------: | :----------: | :------------: |
|   ResNet-50    |           40000            | Crop & Filp  |      73%       |
|   ResNet-50    |            4000            | Crop & Flip  |      48%       |
| ResNet-50(UDA) |            4000            | RandAugment  |      50%       |

-  GPU: (Nvidia RTX 2080-ti)*2
-  ResNet-50 is not pretrained in this experiments.

## Reference
- [[Paper] Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)
