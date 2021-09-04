import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations
import numpy as np
import sys,os
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


class LinearBlock(nn.Module):
    def __init__(self,indim,outdim,classes):
        super(LinearBlock,self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(indim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(indim,outdim)
        )
    
    def forward(self,x):
        return self.block(x)

# class MaskResNet(nn.Module):
#     def __init__(self,name='MaskResNet',classes=18,device='cpu',weight_path='../fixresnet/ResNext101_32x48d_v2.pth'):
#         super(MaskResNet,self).__init__()
#         self.name=name
#         self.device=device
#         self.weight_path = weight_path
#         self.backbone = self.load_FixResNeXt_101_32x48d().to(device)
#         self.classifier = nn.Sequential(
#             LinearBlock(1000,500),
#             LinearBlock(500,200),
#             LinearBlock(200,classes)).to(device)

            
#     def forward(self,x):
#         x = self.backbone(x)
#         return self.classifier(x)
    
#     def load_FixResNeXt_101_32x48d(self):
#         model=resnext_wsl.resnext101_32x48d_wsl(progress=True)
#         pretrained_dict=torch.load(self.weight_path,map_location=self.device)['model']
#         model_dict = model.state_dict()
#         for k in model_dict.keys():
#             if(('module.'+k) in pretrained_dict.keys()):
#                 model_dict[k]=pretrained_dict.get(('module.'+k))
#         model.load_state_dict(model_dict)
#         return model
        

class EfficientB4(nn.Module):
    def __init__(self,name='efficientnet_b4',device='cuda',classes=18):
        super(EfficientB4,self).__init__()
        self.name=name
        self.device=device
        self.classes=classes
        self.backbone = timm.create_model('efficientnet_b4',True).to(device)
        self.classifier = nn.Sequential(
            LinearBlock(1000,500),
            LinearBlock(500,256),
            LinearBlock(256,128),
            LinearBlock(128,64),
            LinearBlock(64,32),
            LinearBlock(32,classes),
        ).to(device)
    
    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)

class EfficientLite0(nn.Module):
    def __init__(self,name='efficientnet_lite0',device='cuda',classes=18):
        super(EfficientB4,self).__init__()
        self.name=name
        self.device=device
        self.classes=classes
        self.backbone = timm.create_model('efficientnet_lite0',True).to(device)
        self.classifier = nn.Sequential(
            LinearBlock(1000,500),
            LinearBlock(500,256),
            LinearBlock(256,128),
            LinearBlock(128,64),
            LinearBlock(64,32),
            LinearBlock(32,classes),
        ).to(device)
    
    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)


class MultiDropoutLinearBlock(nn.Module):
    def __init__(self,indim,outdim,drop_num=5,device='cuda',p=0.5):
        super(MultiDropoutLinearBlock,self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.p=p
        self.linear = nn.Linear(indim,outdim).to(device)
        self.drop_num=drop_num


    def forward(self,x):
        # dr : dropout result
#         x = self.batchnorm(x)
#         x = nn.SiLU()(x)
        dr = None
        for _ in range(self.drop_num):
            out = nn.Dropout(self.p)(x)
            out = self.linear.forward(x)
            if dr is None:
                dr = out
            else:
                dr +=out

        return dr/self.drop_num


class MultiDropoutEfficientB4(nn.Module):
    def __init__(self,name='efficientnet_b4',device='cuda',classes=18,drop_num=10,p=0.8):
        super(MultiDropoutEfficientB4,self).__init__()
        self.name='_'.join(['multi','dropout',name])
        self.device=device
        self.classes=classes
        self.backbone = timm.create_model('efficientnet_b4',True).to(device)
        self.backbone.classifier = MultiDropoutLinearBlock(1792,classes,drop_num=drop_num,p=p).to(device)
#         self.classifier = nn.Sequential(
#             MultiDropoutLinear(1000,500),
#             MultiDropoutLinear(500,256),
# #             MultiDropoutLinear(256,128),
# #             MultiDropoutLinear(128,64),
#             MultiDropoutLinear(256,100),
#             MultiDropoutLinear(100,self.classes)).to(device)
    
    def forward(self,x):
        return self.backbone(x)
        #return self.classifier(x)

class MultiDropoutEfficientLite0(nn.Module):
    def __init__(self,name='efficientnet_lite0',device='cuda',classes=18,drop_num=5,p=0.5):
        super(MultiDropoutEfficientLite0,self).__init__()
        self.name='_'.join(['multi','dropout',name])
        self.device=device
        self.classes=classes
        self.backbone = timm.create_model('efficientnet_lite0',True).to(device)
        self.backbone.classifier = MultiDropoutLinearBlock(1280,classes,drop_num=drop_num,p=p).to(device)
    
    def forward(self,x):
        return self.backbone(x)
        #return self.classifier(x)


