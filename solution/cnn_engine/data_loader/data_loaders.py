import os
import natsort
import numpy as np
import torchvision
from typing import Optional, List, Dict
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.transforms.transforms import CenterCrop, ColorJitter, RandomErasing, RandomRotation, RandomVerticalFlip, Resize
from base import BaseDataLoader
# import transformers
from PIL import Image
import torch
import pandas as pd
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

    Usage data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose(

            [   transforms.Resize((384, 384)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2,saturation=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                transforms.RandomErasing(),
                
            ])

        self.data_dir = data_dir
        self.dataset = UstageMaskDataset(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class UstageMaskDataset(Dataset):
    """[summary]
    Usage dataset
    """
    def __init__(
        self,
        data_dir: str,
        train : bool, 
        download : bool,
        transform : Optional[Compose] = None
    ):
        self.data_dir = data_dir
        
        self.dataframe = pd.read_csv("./pseudo_dataset.csv")
        self.transform = transform

    def get_labels(self):
        return self.dataframe["label"]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx, :]
        path = row["path"]
        
        label = row["label"]
        img_path = path
        image = Image.open(img_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
        
        return image, label