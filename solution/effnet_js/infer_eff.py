from qdnet.models.effnet import Effnet as EfficientModel
import pandas as pd
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from qdnet.conf.config import load_yaml
parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--class_name', help='gender or mask or age')
args = parser.parse_args()
#python infer_eff.py --class_name gender 
#python infer_eff.py --class_name extra_mask
############################################## config
config = load_yaml('./conf/effb3_ns.yaml',args)

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

############################################## 경로!!
if args.class_name == "mask" or "extra_mask":
    out_dim = 3
    paths =["./tf_efficientnet_b3_ns/weight_extra_mask/best_fold0.pth","./tf_efficientnet_b3_ns/weight_extra_mask/best_fold1.pth","./tf_efficientnet_b3_ns/weight_extra_mask/best_fold2.pth", "./tf_efficientnet_b3_ns/weight_extra_mask/best_fold3.pth","./tf_efficientnet_b3_ns/weight_extra_mask/best_fold4.pth"]

elif args.class_name =="gender":
    out_dim = 2
    paths = ["./tf_efficientnet_b3_ns/weight_gender/best_fold0.pth","./tf_efficientnet_b3_ns/weight_gender/best_fold1.pth","./tf_efficientnet_b3_ns/weight_gender/best_fold2.pth", "./tf_efficientnet_b3_ns/weight_gender/best_fold3.pth","./tf_efficientnet_b3_ns/weight_gender/best_fold4.pth"]

elif args.class_name =="age":
#     paths = ["./tf_efficientnet_b3_ns/weight_age_epoch5/best_fold0.pth","./tf_efficientnet_b3_ns/weight_age_epoch5/best_fold1.pth","./tf_efficientnet_b3_ns/weight_age_epoch5/best_fold2.pth", "./tf_efficientnet_b3_ns/weight_age_epoch5/best_fold3.pth","./tf_efficientnet_b3_ns/weight_age_epoch5/best_fold4.pth"]
    paths = ["./tf_efficientnet_b3_ns/weight_age/best_fold0.pth","./tf_efficientnet_b3_ns/weight_age/best_fold1.pth","./tf_efficientnet_b3_ns/weight_age/best_fold2.pth", "./tf_efficientnet_b3_ns/weight_age/best_fold3.pth","./tf_efficientnet_b3_ns/weight_age/best_fold4.pth"]
    
#     paths = ["./tf_efficientnet_b3_ns/weight_age_epoch8/best_fold0.pth","./tf_efficientnet_b3_ns/weight_age_epoch8/best_fold1.pth","./tf_efficientnet_b3_ns/weight_age_epoch8/best_fold2.pth", "./tf_efficientnet_b3_ns/weight_age_epoch8/best_fold3.pth","./tf_efficientnet_b3_ns/weight_age_epoch8/best_fold4.pth"]
    out_dim = 3




N = 12600

##############################################
total_probs = np.zeros((12600, 5))
fin = np.zeros(12600,)
i=0
for path in paths:
    print(path)
    model = EfficientModel(enet_type = config["enet_type"],     
            out_dim = out_dim,         
            drop_nums = int(config["drop_nums"]),
            metric_strategy = config["metric_strategy"]).to(device=device)
    model.load_state_dict(torch.load(path))
    model.eval()
    trsfm = transforms.Compose(
                [   
                    # transforms.Resize((256, 256)),
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    info_dataframe = pd.read_csv("../dummy/eval/info.csv")
    info_dataframe.rename(columns={'ans':args.class_name},inplace=True)


    ans = []

    for idx, row in info_dataframe.iterrows():
        path_evaluation = "../dummy/eval/images"
        path_evaluation = os.path.join(path_evaluation, row["ImageID"])
        img = Image.open(path_evaluation)
        img = trsfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(img)
            probs = F.softmax(probs,dim =1)
            probs = probs.cpu().detach().numpy()
            probs = probs.argmax(1)
            total_probs[idx, i] = probs
#             print(total_probs[idx])
        ans.append(probs)
        if idx % 100 == 0:
            print(idx, probs)


    i+=1
#     print(total_probs)
f=0
origin_prob = pd.DataFrame(columns=['0','1','2','3','4'])

for i in total_probs:
    
    origin_prob = origin_prob.append(list(i))
    i = i.astype(int)
    a = np.bincount(i)
    pre = np.argmax(a)
    fin[f]=pre
    f+=1


info_dataframe["ans"] = fin
print(info_dataframe.head())
info_dataframe.to_csv(f"./{args.class_name}_submission.csv", index=False)
origin_prob.to_csv(f"./{args.class_name}_before_vote.csv", index=False)
print("DONE")