import os
import natsort
import numpy as np
import torchvision
from typing import Optional, List, Dict, Union
from torch.utils.data import Dataset
import transformers
from PIL import Image
from copy import deepcopy


Compose = torchvision.transforms.transforms.Compose
FeatureExtractor = transformers.feature_extraction_utils.FeatureExtractionMixin


def shuffle_idx(*args, seed=42):
    length = set(map(len, args))
    assert len(length) == 1
    idx = [i for i in range(list(length)[0])]
    np.random.shuffle(idx)
    args = [[arg[i] for i in idx] for arg in args]
    return args


def select(dataset, ratio, right=False):
    new_dataset = deepcopy(dataset)
    start = None if right else int(len(new_dataset)*ratio)
    end = None if not right else int(len(new_dataset)*ratio)
    new_dataset.total_imgs = new_dataset.total_imgs[start:end]
    new_dataset.labels = new_dataset.labels[start:end]
    return new_dataset


def train_test_split(dataset, test_size=0.2, shuffle=True, seed=42):
    assert test_size > 0 and test_size < 1
    new_dataset = deepcopy(dataset)
    if shuffle:
        x, y = shuffle_idx(new_dataset.total_imgs, new_dataset.labels, seed=42)
        new_dataset.total_imgs = x
        new_dataset.labels = y
    train_dataset = select(new_dataset, 1 - test_size, right=True)
    test_dataset = select(new_dataset, 1 - test_size, right=False)
    return train_dataset, test_dataset


class FileReadMixin:

    @classmethod
    def load(cls, data_dir_or_file, is_train=False, **kwargs):
        if isinstance(data_dir_or_file, str):
            imgs, labels = cls.read(data_dir_or_file, is_train)
        else:
            raise NotImplemented

        return FaceMaskDataset(
            data_dir=data_dir_or_file,
            total_imgs=imgs,
            labels=labels,
            is_train=is_train,
            **kwargs,
        )

    @classmethod
    def read(cls, data_dir: str, is_train: bool = False):
        all_imgs = os.listdir(data_dir)
        all_imgs_file_names = []
        labels = [] if is_train else None
        for img in all_imgs:
            if is_train:
                subfolder = os.path.join(data_dir, img)
                if os.path.isdir(subfolder):
                    for file_name in os.listdir(subfolder):
                        if not file_name.startswith("._"):
                            file_path = os.path.join(img, file_name)
                            all_imgs_file_names.append(file_path)
                            labels.append(cls.get_label_from_filename(img, file_name))
            else:
                if not img.startswith("._"):
                    all_imgs_file_names.append(img)
        return all_imgs_file_names, labels

    @staticmethod
    def add_file(file_list, file_name):
        if not file_name.startswith("._"):
            file_list.append(file_name)

    @staticmethod
    def get_label_from_filename(folder, file_name):
        # mask label
        if file_name.startswith("incorrect"):
            mask = "Incorrect"
        elif file_name.startswith("normal"):
            mask = "Not Wear"
        elif file_name.startswith("mask"):
            mask = "Wear"
        else:
            raise ValueError(f"{file_name}")
        # gender label
        gender = folder.split("_")[1].title()
        # age label
        age = int(folder.split("_")[-1])
        if age < 30:
            age = "<30"
        elif age < 60:
            age = ">=30 and <60"
        else:
            age = ">=60"
        return [mask, gender, age]


class FaceMaskDataset(Dataset, FileReadMixin):

    mask2id = {"Wear": 0, "Incorrect": 1, "Not Wear": 2}
    gender2id = {"Male": 0, "Female": 1}
    age2id = {"<30": 0, ">=30 and <60": 1, ">=60": 2}

    def __init__(
        self,
        data_dir: str,
        total_imgs: List[str],
        labels: List[Union[int, List[int]]],
        transform: Union[Compose, FeatureExtractor],
        return_image: bool = True,
        level: int = 1,
        is_train: bool = False,
    ):
        assert level in [1, 3]
        self.level = level
        self.data_dir = data_dir
        self.transform = transform
        self.total_imgs = total_imgs
        self.labels = labels
        self.return_image = return_image
        self.is_train = is_train

    def select(self, ratio=0.1, right=False):
        return select(self, ratio=ratio, right=right)

    def train_test_split(self, test_size=0.2, shuffle=True, seed=42):
        return train_test_split(self, test_size=test_size, shuffle=shuffle, seed=seed)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.data_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        pixel_values = self.transform(image)
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        elif isinstance(pixel_values, dict):
            pixel_values = pixel_values["pixel_values"][0]
        output = {"pixel_values": pixel_values}
        if self.is_train:
            mask = self.mask2id[self.labels[idx][0]]
            gender = self.gender2id[self.labels[idx][1]]
            age = self.age2id[self.labels[idx][2]]
            label = [mask, gender, age]
            if self.level == 1:
                label = 6 * mask + 3 * gender + age
            output.update({"label": label})
        if self.return_image:
            output.update({"image": Image.open(img_loc).convert("RGB")})
        return output
