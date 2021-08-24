import os
import natsort
import numpy as np
import torchvision
from typing import Optional, List, Dict
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop
from base import BaseDataLoader
from PIL import Image
import torch
import pandas as pd

Compose = torchvision.transforms.transforms.Compose


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CifarDataLoader(BaseDataLoader):
    """
    Cifar data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class UstageDataLoader(BaseDataLoader):
    """Contest Data Loader

    Args:
        BaseDataLoader ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose(
            [   transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.data_dir = data_dir
        self.dataset = UstageMaskDataset(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class UstageMaskDataset(Dataset):
    """[summary]

    Args:
        Dataset ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    def __init__(
        self,
        data_dir: str,
        train : bool,
        transform : Optional[Compose] = None,
        level: int = 1,
    ):
        assert level in [1, 3], "level must be 1 or 3."
        self.data_dir = data_dir
        self.dataframe = pd.read_csv("./labeled_train.csv")
        self.transform = transform
        self.level = level

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx, :]
        path = row["path"]

        label_mask, label_gender, label_age = row["mask"], row["gender"], row["age"]

        if self.level == 1:
            label = label_mask * 6 + label_gender * 3 + label_age
        elif self.level == 3:
            label = [label_mask, label_gender, label_age]
        else:
            raise AttributeError(f"level must be 1 or 3. but got {level}.")

        img_path = os.path.join("..", path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
