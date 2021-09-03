### This is a repository of boostcamp image classification contest - level 1

- The classification model is divided into 3 models. Age, gender, mask.
- The results are then concatted into 1 submission csv which contains total 18 classes(3*3*2)


## 0.Data format transform

```
python tools/datapreprocess.py
```

- Divide the data into specific folds in the config file
- Creates total 3 csvs (age, gender, mask)


## 1.TTS
```
python train_test_split.py 
```

## 2.train
```
python train.py --config_path "conf/effb3_ns.yaml"
```

## 3.test
```
python test.py --config_path "conf/effb3_ns.yaml" --n_splits 5

```

## 4.inference
```
python infer_eff.py

```

## 5. Make submission File
```
python all_in_one.py

```



## Model : effb3_ns
1、Modify configuration file

```
data_dir : #train/test csv
data_folder : #train/test csv
image_size : #image size
enet_type : "tf_efficientnet_b3_ns" 
batch_size : 16
num_workers : 5
init_lr : 3e-5
out_dim : # number of classes
n_epochs : 4
drop_nums : 1
loss_type : "focal_loss"  # ce_loss, ce_smothing_loss, focal_loss, bce_loss, mlsm_loss
mixup_cutmix : False
model_dir : #path to save model in 
log_dir : #path to save log in
CUDA_VISIBLE_DEVICES : "0"  
fold : "0,1,2,3,4"
pretrained : True
eval : "best"                  # "best", "final"
oof_dir : #test result directory "./tf_efficientnet_b3_ns/oofs/"
auc_index : 0
```



#

#

#

#

#

#

#

#### ref
```
（1）https://github.com/MachineLP/PyTorch_image_classifier






