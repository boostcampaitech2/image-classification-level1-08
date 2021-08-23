import os
import natsort
import numpy as np
import torchvision
from typing import Optional, List, Dict
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from base import BaseDataLoader


Compose = torchvision.transforms.transforms.Compose
FeaturExtractor = transformers.feature_extraction_utils.FeatureExtractionMixin


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


class BaseDatasetForThreeHead(Dataset):
    """
    @jinmang2 21.08.23
    3개의 Classification Head로 분류하는 모델을 위한 Dataset base class
    """

    mask2id = {
        "Wear": 0,
        "Incorrect": 1,
        "Not Wear": 2,
    }

    gender2id = {
        "Male": 0,
        "Female": 1,
    }

    age2id = {
        "<30": 0,
        ">=30 and <60": 1,
        ">=60": 2,
    }

    def __init__(
        self,
        data_dir: str,
        feature_extractor: Optional[FeaturExtractor] = nn.Identity(),
        transform: Optional[Compose] = None,
    ):
        """
        Base Dataset 객체 생성자!

        Args
        ======
            data_dir: data가 있는 root directory
            feature_extractor: feature 추출 모델. 입력하지 않을 경우 identity mapping
            transform: 입력 이미지 전처리 및 Tensor로 변환
        """
        self.data_dir = data_dir
        self.transform = transform
        self.feature_extractor = feature_extractor
        all_imgs = os.listdir(data_dir)
        # image file name을 가져오자!
        all_imgs_file_names = []
        labels = []
        for img_path in all_imgs:
            # 폴더인 경우에만 서치
            subfolder = os.path.join(data_dir, img_path)
            if os.path.isdir(subfolder):
                for file_name in os.listdir(subfolder):
                    # 파일이 온전한 jpg인 경우만 서치
                    if not file_name.startswith("._"):
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
                        gender = img_path.split("_")[1].title()
                        # age label
                        age = int(img_path.split("_")[-1])
                        if age < 30:
                            age = "<30"
                        elif age < 60:
                            age = ">=30 and <60"
                        else:
                            age = ">=60"
                        all_imgs_file_names.append(
                            os.path.join(img_path, file_name)
                        )
                        labels.append(
                            [mask, gender, age]
                        )
        self.total_imgs = all_imgs_file_names
        self.labels = labels

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        학습/평가 데이터셋을 구분!

        Args:
            test_size: test set의 크기 비율 (0 ~ 1 사이의 실수)
            shuffle: imgs file path와 label을 섞어줄지 여부

        Returns:
            (BaseDatasetForThreeHead, BaseDatasetForThreeHead)
        """
        assert test_size > 0 and test_size < 1
        train_dataset = CustomDataset(
            self.data_dir, self.feature_extractor, self.transform)
        test_dataset = CustomDataset(
            self.data_dir, self.feature_extractor, self.transform)
        if shuffle:
            total_imgs, labels = self.shuffle()
        else:
            total_imgs, labels = self.total_imgs, self.labels
        setattr(train_dataset, "total_imgs", total_imgs[:int(len(self) * (1-test_size))])
        setattr(test_dataset, "total_imgs", total_imgs[int(len(self) * (1-test_size)):])
        return train_dataset, test_dataset

    def shuffle(self, seed=42):
        """
        주어진 seed number로 섞인 img file path와 labels를 반환
        """
        idx = [i for i in range(len(self))]
        np.random.shuffle(idx)
        total_imgs = [self.total_imgs[i] for i in idx]
        labels = [self.labels[i] for i in idx]
        return total_imgs, labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.data_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)
        pixel_values = self.feature_extractor(image)["pixel_values"][0]
        labels =[
            self.mask2id[self.labels[idx][0]],
            self.gender2id[self.labels[idx][1]],
            self.age2id[self.labels[idx][2]],
        ]
        return {
            "pixel_values": torch.from_numpy(pixel_values),
            "label": labels,
#             "image": image,
        }
