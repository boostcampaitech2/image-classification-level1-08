import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.loss
from sklearn.metrics import f1_score
import sklearn
import numpy as np
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

class SoftTargetCrossEntropy(nn.Module):
    def __init__(SoftTargetCrossEntropy,self):
        self.loss_fn = timm.loss()
    
    def forward(self,y_pred,y_true):
        return self.loss_fn(y_pred,y_true)


class F1Loss(nn.Module):
    def __init__(self,is_training=True):
        super(F1Loss,self).__init__()
        self.is_training=is_training
        
    def forward(self,y_pred,y_true):
        assert y_true.ndim == 1
        assert y_pred.ndim == 1 or y_pred.ndim == 2

        if y_pred.ndim == 2:
            y_pred = y_pred.argmax(dim=1)

        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        epsilon = 1e-7

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f1 = 2* (precision*recall) / (precision + recall + epsilon)
        f1.requires_grad = self.is_training
        return 1 -f1
        
    

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=18, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )