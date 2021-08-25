import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import torchvision
import clip
from efficientnet_pytorch import EfficientNet


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CifarModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class ResNet18(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return F.log_softmax(x, dim=1)

class ResNet34(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet34(pretrained=True)
        self.feature_extractor.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        return F.log_softmax(x, dim=1)

class ClipModel(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.fc1 = nn.Linear(512, 256)
        self.batch = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.encode_image(x)
        x = x.float()

        x = self.fc1(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class EfficientModel(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.feature_extractor = EfficientNet.from_pretrained("efficientnet-b5")
        self.fc = nn.Linear(1000, num_classes)
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class ClipThreeHeadModel(BaseModel):
    def __init__(self, num_classes=18):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32")

        self.fc = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 3)
        self.fc2 = nn.Linear(256, 2)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.encode_image(x)
        x = x.float()
        x = self.relu(self.fc(x))
        return F.log_softmax(self.fc1(x), dim=-1), F.log_softmax(self.fc2(x), dim=-1), F.log_softmax(self.fc3(x), dim=-1)