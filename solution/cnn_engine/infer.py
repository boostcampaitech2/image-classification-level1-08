from model.model import ClipModel, EfficientModel, Efficientv2Model, NfnetModel
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

device = "cuda" if torch.cuda.is_available() else "cpu"

paths = ["./saved/models/Nfnet/0902_060410/model_best.pth"]
N = 12600
total_probs = np.zeros((12600, 18))
for path in paths:
    print(path)
    model = NfnetModel().to(device=device)
    model.load_state_dict(copyStateDict(torch.load(path)["state_dict"]))
    model.eval()
    trsfm = transforms.Compose(
                [   
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    info_dataframe = pd.read_csv("../input/data/eval/info.csv")
    print(info_dataframe.head())

    ans = []

    for idx, row in info_dataframe.iterrows():
        path_evaluation = "../input/data/eval/images"
        path_evaluation = os.path.join(path_evaluation, row["ImageID"])
        img = Image.open(path_evaluation).convert('RGB')
        img = trsfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = (model(img) + model(torch.flip(img, dims=[3]))) / 2
            probs= probs.cpu().numpy()
            total_probs[idx, :] += probs[0]
            prediction = np.argmax(probs[0], axis=-1)
        ans.append(prediction)
        if idx % 100 == 0:
            print(idx, prediction)


info_dataframe["ans"] = np.argmax(total_probs, axis=-1)
info_dataframe.to_csv("./submission.csv", index=False)