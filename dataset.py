from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def Dataset(Dataset):  ##
    def __init__(self, path: str):
        pass

    def __len__(self) -> int:
        return

    def __getitem__(self, idx):
        return


def get_dataloader(path: str, batch_size: int) -> Tuple[DataLoader]:  ##
    """
    TODO: add collate function or transformation

    :return: train_loader, val_loader, test_loader
    """
    return
