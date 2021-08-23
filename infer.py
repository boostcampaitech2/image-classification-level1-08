from model.model import ResNet18
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


model = ResNet18(num_classes=18)
model.load_state_dict(copyStateDict(torch.load("./saved/models/Ustage_CLIP/0823_221619/model_best.pth")["state_dict"]))

trsfm = transforms.Compose(
            [   transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

info_dataframe = pd.read_csv("../input/data/eval/info.csv")
print(info_dataframe.head())

ans = []

for idx, row in info_dataframe.iterrows():
    path = "../input/data/eval/images"
    path = os.path.join(path, row["ImageID"])
    # print(path)
    img = Image.open(path)
    img = trsfm(img).unsqueeze(0)
    with torch.no_grad():
        probs = model(img).cpu().numpy()
        prediction = np.argmax(probs[0], axis=-1)
    ans.append(prediction)
    print(f"{idx} : Label probs:", prediction) 

info_dataframe["ans"] = ans
info_dataframe.to_csv("./submission.csv", index=False)