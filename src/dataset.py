import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from randaugment import RandAugment  # https://github.com/jizongFox/pytorch-randaugment
from torch.utils.data import DataLoader, Dataset


class CIFAR10(Dataset):
    """
    Custom CIFAR-10 dataset for semi-supervised setting.

    :param path: root path of CIFAR-10 dataset
    :param mode: train/val/test
    :param labeled: if false, only return unlabeled data
    :param uda: whether to apply uda
    :param data_batch: number of data batches of CIFAR-10 (default: 5)
    :param num_labeled: number of labeled data to use (must-use in uda)
    """

    def __init__(
        self,
        path: str,
        mode: str,
        labeled: bool,
        uda: bool,
        data_batch: int = 5,
        num_labeled: int = None,
    ):
        super(CIFAR10, self).__init__()
        assert mode in ["train", "val", "test"], "You must choose train/val/test"
        self.labeled = labeled
        self.uda = uda

        self.images = None
        self.labels = []

        if mode in ["train", "val"]:
            # aggregate data batches
            for batch in range(data_batch):
                data = None
                with open(os.path.join(path, f"data_batch_{batch+1}"), "rb") as fo:
                    data = pickle.load(fo, encoding="bytes")
                img = data[b"data"]
                img = img.reshape(len(img), 32, 32, 3)
                label = data[b"labels"]

                try:
                    self.images = np.concatenate((self.images, img), axis=0)
                except ValueError:  ## at first batch
                    self.images = img
                self.labels.extend(label)

            if not uda:
                assert (
                    labeled
                ), "You cannot use unlabeled data when not using UDA. Please set labeled==True."

                # convert data type and split train/val
                n_train = 40000
                self.images = self.images[:n_train] if mode == "train" else self.images[n_train:]
                self.labels = (
                    torch.LongTensor(self.labels)[:n_train]
                    if mode == "train"
                    else torch.LongTensor(self.labels)[n_train:]
                )
                assert len(self.images) == len(self.labels), "data/label length mismatch"

                # You can set the number of labeled data to use
                if mode == "train" and num_labeled is not None:
                    assert num_labeled <= n_train, f"labeled data cannot be over {n_train}."
                    self.images = self.images[:num_labeled]
                    self.labels = self.labels[:num_labeled]

            else:  # uda
                n_train, n_val = 40000, 10000
                assert num_labeled is not None, "You must set num_labeled value in UDA."
                assert num_labeled <= n_train, f"labeled data cannot be over {n_train}."

                if mode == "train":
                    self.images = (
                        self.images[:num_labeled] if labeled else self.images[num_labeled:-n_val]
                    )
                    self.labels = (
                        self.labels[:num_labeled] if labeled else self.labels[num_labeled:-n_val]
                    )
                else:
                    self.images = self.images[n_train:]
                    self.labels = self.labels[n_train:]
                self.labels = torch.LongTensor(self.labels)

        else:  # test
            data = None
            with open(os.path.join(path, "test_batch"), "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
            self.images = data[b"data"]
            self.images = self.images.reshape(len(self.images), 32, 32, 3)
            self.labels = torch.LongTensor(data[b"labels"])

        print(f"[{mode.upper()}] Number of data: {len(self.images)}")
        print(f" ** Current Options ** UDA: {self.uda} / Labeled: {self.labeled}")

    def augmentation(self, img, randaugment=False):
        if randaugment:
            img = Image.fromarray(img)
            return transforms.Compose(
                [
                    RandAugment(),
                    transforms.ToTensor(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )(img)
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )(img)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_labeled = self.augmentation(image, randaugment=False)
        label = self.labels[idx]

        if self.uda and not self.labeled:  # for uda dataloader
            image_unlabeled = self.augmentation(image, randaugment=False)
            image_augmented = self.augmentation(image, randaugment=True)
            return {"unlabeled": image_unlabeled, "augmented": image_augmented}

        return {"labeled": image_labeled, "label": label}


def get_dataloader(
    path: str,
    mode: str,
    batch_size: int,
    num_labeled=None,
    labeled: bool = True,
    uda: bool = False,
) -> DataLoader:
    """
    :param path: root path of CIFAR-10 dataset
    :param mode: train/val/test
    :param batch_size: mini-batch size
    :param num_labeled: number of labeled data to use (must-use in uda)
    :param labeled: if false, only return unlabeled data (only for train data)
    :param uda: whether to apply uda

    :return: dataloader
    """
    dset = CIFAR10(path, mode=mode, num_labeled=num_labeled, labeled=labeled, uda=uda)
    shuffle = True if mode == "train" else False
    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
