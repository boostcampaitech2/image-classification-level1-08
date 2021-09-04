# AI Boostcamp P Stage Mask Classification
This codes is about to AI Boostcamp P stage Mask Classification competition.  
Below are code usage and My competitions review.  

# Requirements
```
albumentations==1.0.3
numpy==1.19.2
timm==0.4.12
torch==1.7.1
torchvision==0.8.2
tqdm==4.51.0
ttach==0.0.3
wandb==0.12.1
```
You can install requirements by typing  `pip install -r requirements.txt`

```
+-- train/
|   +-- images/
|       +-- 000001_female_Asian_45/
|       +-- 000002_female_Asian_52/
|       +-- …
|   +-- standard.csv
+-- eval/
    +-- images/
        +-- 814bff668ae5b9c595ceabcbb6e1ea84634afbd6.jpg
        +-- 819f47db0617b3ea9725ef1f6f58e56561e7cb4b.jpg
        +-- …
    +-- info.csv
```
You must have **mask dataset** and this **data directory structure** before execution.

# Usage
**BASIC**

`python3 train.py -config config.json`

**OPTION**
```
--seed : random seed default: 1010, type=int, default=1010
--epochs : number of epochs to train, type=int, default=5
--dataset : dataset augmentation type default: MaskDataSet, type=str, default=MaskDataSet
--augmentation : data augmentation type default: MaskAugmentation, type=str, default=MaskAugmentation
--resize, resize size for image when training, type=list, default=[256, 256]
--batch_size : batch size for training,  type=int, default=128
--kfold_splits : k-fold splits number for cross validation, type=int, default=5
--model : model type for learning, type=str, default: MultiDropoutEfficientLite0
--optimizer : optimizer type, type=str, default=AdamW
--lr : learning rate, type=float, default=1e-2
--criterion : criterion type, type=str, default=FocalLoss
--weight_decay : Weight decay for optimizer, type=int, default=1e-4, 
--lr_scheduler : learning scheduler, type=str, default=OneCycleLR
--log_interval : how many batches to wait before logging training status type=int, default=30 
--file_name : model save at {results}/{file_name} default=exp
--train_csv : train data saved csv, default=/opt/ml/input/data/train/new_standard.csv 
--test_dir : test data saved directory, default="/opt/ml/input/data/eval" 
--mix_up : if True, mix-up & cut-mix use, type=boolean_string, default=False
--num_class : input the number of class, type=int,default=18
--pseudo_label : pseudo label usage, Should write pseudo.csv location at pseudo_csv option, type=boolean_string, default=False
--pseudo_csv : pseudo label usage, type=str, default=/opt/ml/input/data/train/pseudo.csv
--wandb : logging in WandB, type=boolean_string, default=True
--patience : early stopping patience number, type=Int, default=5
```

- ex. `python3 train.py --config config.json --epochs 30 --file_name my_test_net --mix_up True --seed 42 --pseudo_label True --wandb False`

- **cause** : You have to download new_standard.csv and write file location at option `--train_csv`

- Detail configuration controll(Optimizer, Scheduler, ... etc) is updated as soon as possible
