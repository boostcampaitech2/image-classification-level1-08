import torch
import torchvision
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import os
import albumentations
import albumentations.pytorch
import ttach as tta
import os
import random

def seed_everything(seed):
    ''' Fix Random Seed '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(1010)


class MaskAugmentation:
    def __init__(self,image_size=[256,256]):
        self.transforms={
            # 'train': albumentations.Compose([
            #                 albumentations.Resize(image_size[0],image_size[1]),
            #                 #albumentations.RandomCrop(224,224),
            #                 albumentations.CenterCrop(int(image_size[0]*0.875),int(image_size[1]*0.875)),
            #                 albumentations.ColorJitter(),
            #                 albumentations.HorizontalFlip(),
            #                 albumentations.Normalize(),
            #                 albumentations.pytorch.ToTensorV2(),
            #                ]),
            'train' : albumentations.Compose([
                            albumentations.Resize(image_size[0],image_size[1]),
                            albumentations.CenterCrop(224,224),
                            albumentations.ColorJitter(),
                            albumentations.Normalize(),
                            albumentations.GaussNoise(var_limit=(0.001,0.01)),
                            albumentations.CoarseDropout(max_height=25,max_width=25),
                            albumentations.RandomBrightnessContrast(),
                            albumentations.Rotate(15),
                            albumentations.HorizontalFlip(),
                            albumentations.pytorch.ToTensorV2()]),

            'validation' : albumentations.Compose([
                            albumentations.Resize(224,224),
                            albumentations.Normalize(),
                            albumentations.pytorch.ToTensorV2()]),
                            
            'tta' : tta.Compose([
                            tta.HorizontalFlip()
            ])
        }
    
    def __call__(self,mode='train'):
        if not mode in self.transforms.keys():
            raise ValueError(f'mode have to be {list(self.transforms.keys())}')
        return self.transforms[mode]

class MaskDataSet(torch.utils.data.Dataset):
  
    def __init__(self,train_csv='/opt/ml/input/data/train/new_standard.csv',multi_label=False,transform=None):
        self.transform=transform
        self.multi_label=multi_label

        self.image_paths, self.label_classes, self.labels = self.read_csv_file(train_csv)
        
    def __len__(self):
        return len(self.image_paths)    
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image data 불러오기
        image = np.array(Image.open(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.multi_label:
            y = np.array(self.labels[idx])
        else:
            y = np.array(self.label_classes[idx])
        
        return image,y
        
    def set_transform(self,transform):
        self.transform=transform
    
    def read_csv_file(self,train_dir):
        ''' Return file path using directory path in csv_file  '''
        data_pd = pd.read_csv(train_dir,encoding='utf-8')

        return data_pd['path'], data_pd['label'], list(zip(data_pd['gender'],data_pd['age'],data_pd['mask']))


class PseudoMaskDataSet(torch.utils.data.Dataset):
  
    def __init__(self,train_csv='/opt/ml/input/data/train/new_standard.csv',
    pseudo_csv='/opt/ml/input/data/train/pseudo.csv'
    ,multi_label=False,transform=None):
        assert multi_label==False, "multi label is not used pseudo labeling"
        self.transform=transform
        self.image_paths, self.label_classes = self.read_csv_file(train_csv,pseudo_csv)
        
    def __len__(self):
        return len(self.image_paths)    
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # image data 불러오기
        image = np.array(Image.open(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image=image)['image']
    
        y = np.array(self.label_classes[idx])
        
        return image,y
        
    def set_transform(self,transform):
        self.transform=transform
    
    def read_csv_file(self,train_csv,pseudo_csv):
        ''' Return file path using directory path in csv_file  '''
        data_pd = pd.read_csv(train_csv,encoding='utf-8')
        pseudo_pd = pd.read_csv(pseudo_csv,encoding='utf-8')
        
        path = pd.concat([pseudo_pd['ImageID'].apply(lambda x: os.path.join('/opt/ml/input/data/eval/images',x)),data_pd['path']]).reset_index(drop=True)
        label = pd.concat([pseudo_pd['ans'],data_pd['label']]).reset_index(drop=True)
        
        return path, label
                 
                 

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,test_dir,transform=albumentations.Resize(224,224)):
        self.transform = transform
        self.submission = pd.read_csv(os.path.join(test_dir,'info.csv'))
        
        image_dir = os.path.join(test_dir,'images')
        self.image_paths = [os.path.join(image_dir,image_id) for image_id in self.submission.ImageID]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = np.array(Image.open(self.image_paths[idx]))
        
        if self.transform:
            image = self.transform(image=image)['image'].float()

        return image

class CustomSubset(torch.utils.data.Subset):

    def __init__(self, dataset, indices,transform=None):
        super(CustomSubset,self).__init__(dataset,indices)
        self.dataset = dataset
        self.indices = indices
        self.transform=transform

    def __getitem__(self, idx):
        x,y = self.dataset[self.indices[idx]]
        x = self.transform(image=np.array(x))['image'].float()
        return x,y

    def __len__(self):
        return len(self.indices)