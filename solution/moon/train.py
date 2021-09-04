import wandb
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import sys
import pathlib
from tqdm import tqdm
import random
import timm
import timm.loss
import albumentations
import albumentations.pytorch
#!pip install GPUtil
import GPUtil
from sklearn.metrics import f1_score
import warnings
from sklearn.model_selection import StratifiedKFold
import ttach as tta

# custom module
import model
import dataset
from dataset import MaskAugmentation
import loss

warnings.filterwarnings('ignore')


def seed_everything(seed):
    ''' Fix Random Seed '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(config):
    ''' Train model and Inference Test data '''
    
    print('='*100)
    print('start...')
    print(config)
    ## Random seed configuration
    seed_everything(config['seed'])

    ## device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device :[{device}]')

    ## transforms configuration
    transforms = MaskAugmentation(image_size=config['resize'])
    train_trsfm = transforms('train')
    val_trsfm = transforms('validation')
    tta_trsfm = transforms('tta')

    # Stratified K-Fold configuration
    n_splits = config['kfold_splits']
    s_kfold = StratifiedKFold(n_splits=n_splits)

    # HyperParameter & Constant Loading
    LEARNING_RATE = config['lr']
    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    PRINT = config['log_interval']
    MIX_UP = config['mix_up']
    CLASSES = config['num_class']

    # dataset configuration
    if config['pseudo_label']:
        mask_dataset = dataset.PseudoMaskDataSet(config['train_csv'],config['pseudo_csv'])
    else:
        mask_dataset = dataset.MaskDataSet(config['train_csv'])
    
    test_dataset = dataset.TestDataset(config['test_dir'],transform=val_trsfm)
    submission = test_dataset.submission
    test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=False)
    # cut-mix, mix-up configuration
    ''' [수정] mixup_args config json'''
    if MIX_UP:
        mixup_args = {
        'mixup_alpha': 0.,
        'cutmix_alpha': 0.3,
        'cutmix_minmax': None,
        'prob': 0.5,
        'switch_prob': 0.,
        'mode': 'elem',
        'label_smoothing': 0.1,
        'num_classes': 18}
        mixup_fn = timm.data.mixup.Mixup(**mixup_args)

    # results directory configuration
    file_name = config['file_name']
    os.makedirs(os.path.join(os.getcwd(),'results',file_name),exist_ok=True)
    
    # OOF configuration
    off_pred = None # Out-Of-Fold predictions
    
    ## training
    for i,(train_idx,val_idx) in enumerate(s_kfold.split\
        (mask_dataset.image_paths,mask_dataset.label_classes)):
        # Dataset Define
        train_set = dataset.CustomSubset(mask_dataset,train_idx,train_trsfm)
        val_set = dataset.CustomSubset(mask_dataset,val_idx,val_trsfm)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,\
             shuffle=True, num_workers=4, drop_last=True, pin_memory=torch.cuda.is_available())
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE,\
            shuffle=True, num_workers=4, drop_last=True, pin_memory=torch.cuda.is_available())
        
        # Model Define
        ''' [수정] model config json 추가!!'''
        net = getattr(model,config['model'])(device=device) 
        print('='*100)
        print(f"{i+1}_model : [{net.name}]")

        # Criterion, optimizer, scheduler
        if MIX_UP:
            criterion = timm.loss.SoftTargetCrossEntropy()
        else:
            criterion = getattr(loss,config['criterion'])()
        ''' [수정] optimzier config json 추가 '''
        optm = getattr(optim,config['optimizer'])(net.parameters(),lr=LEARNING_RATE)
        ''' [수정] lr_scheduler config json 추가 '''
        scheduler = optim.lr_scheduler.OneCycleLR(optm,max_lr=0.1, epochs=EPOCHS, steps_per_epoch = len(train_loader))

        # for Logging(Wandb)
        if config['wandb']:
            wandb.init(config=config,entity='larcane',project='ai_boostcamp_p_stage_mask_classification',
                group=config['file_name'],name=f"{i+1}_model")
            wandb.watch(net)

        # for early stopping
        patience = config['patience']
        counter=0
        # for saving best model parameter
        best_val_loss = np.inf

        for epoch in range(EPOCHS):
            torch.cuda.empty_cache()
            net.train()
            # for logging
            loss_list = []
            f1_list= []
            acc_items_list = [0,0]
            for idx,(batch_in, batch_out) in enumerate(train_loader):
                # # mix-up
                batch_in, batch_out = batch_in.to(device), batch_out.to(device)
                if MIX_UP:
                    batch_in, batch_out = mixup_fn(batch_in,batch_out)

                y_pred = net.forward(batch_in)
                y_true = batch_out # for renaming

                # loss calc
                loss_out = criterion(y_pred,y_true)
                loss_list.append(loss_out.item())

                y_pred = y_pred.argmax(-1)
                if MIX_UP:
                    y_true = y_true.argmax(-1)
                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()

                # accuracy calc
                acc_items_list[0] +=(y_pred==y_true).sum().item() # correct num
                acc_items_list[1] +=len(y_pred) # total num
                # f1 score calc
                f1_list.append(f1_score(y_true,y_pred,average='macro').item())
                
                if (idx+1) % PRINT==0:
                    train_acc = float(acc_items_list[0]/acc_items_list[1])
                    train_f1 = np.mean(f1_list)
                    train_loss = np.mean(loss_list)
                    print(
                    f"Epoch[{epoch+1:4}/{EPOCHS:4}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || f1 score {train_f1:.2f}")

                    ## logging by wandb
                    if config['wandb']:
                        wandb.log({'train_loss':train_loss, 'train_accuracy':train_acc, 'train_f1':train_f1})
                    

                    loss_list = []
                    f1_list= []
                    acc_items_list = [0,0]

                ''' [수정] gradient Accumulation 추가 '''
                optm.zero_grad()
                loss_out.backward()
                optm.step()

            scheduler.step()

            # validation_set test
            net.eval()
            with torch.no_grad():
                loss_list = []
                f1_list= []
                acc_items_list = [0,0]

                for batch_in, batch_out in val_loader:
                    batch_in, batch_out = batch_in.to(device), batch_out.to(device)

                    y_pred = net.forward(batch_in)
                    y_true = batch_out # for renaming
                    if MIX_UP:
                        y_true = F.one_hot(y_true,CLASSES)

                    # loss calc
                    loss_list.append(criterion(y_pred,y_true).item())

                    y_pred = y_pred.argmax(-1)
                    if MIX_UP:
                        y_true = y_true.argmax(-1)
                    y_pred = y_pred.detach().cpu().numpy()
                    y_true = y_true.detach().cpu().numpy()

                    # accuracy calc
                    acc_items_list[0] +=(y_pred==y_true).sum().item() # correct num
                    acc_items_list[1] +=len(y_pred) # total num
                    # f1 score calc
                    f1_list.append(f1_score(y_true,y_pred,average='macro').item())
                
                val_acc = float(acc_items_list[0]/acc_items_list[1])
                val_f1 = np.mean(f1_list)
                val_loss = np.mean(loss_list)
                # logging by wandb
                if config['wandb']:
                    wandb.log({'validation_loss':val_loss, 'valication_accuracy':val_acc, 'validation_f1':val_f1})

                ## Callback function : earuly stopping & best f1 score model saving
                if best_val_loss > val_loss:
                    print(f"New best validation loss !, save {i+1}_model in results/{file_name}")
                    best_val_loss = val_loss
                    torch.save(net.state_dict(),f"results/{file_name}/{i+1}_{epoch:03}_f1_{val_f1:4.2f}.pt")
                    counter=0
                else:
                    counter+=1
                print(f"validation_loss {val_loss:4.4} || valication_accuracy {val_acc:4.2%} || validation_f1 {val_f1:.2f}")
                ## early stopping
                if counter > patience:
                    print(f'Early Stopping{i+1}_{epoch:03}...')
                    break
        
        # Test_set Test
        net.eval()
        test_predictions = []
        with torch.no_grad():
            print('='*100)
            print('Testing..')
            for image in test_loader:
                image = image.to(device)

                # Test Time Augmentation
                tta_pred=None
                for transformer in tta_trsfm:
                    t_image = transformer.augment_image(image)
                    y_pred = net.forward(t_image)

                    if tta_pred is None:
                        tta_pred = y_pred
                    else:
                        tta_pred +=y_pred
                
                pred = (tta_pred/len(tta_trsfm)).detach().cpu().numpy()
                test_predictions.extend(pred)     
            fold_pred = np.array(test_predictions)
            with open(f'./results/{file_name}/{i+1}_logits.pkl','wb') as f:
                pickle.dump(fold_pred,f)

        if off_pred is None:
            off_pred = fold_pred/n_splits
        else:
            off_pred += fold_pred/n_splits
    
    submission['ans'] = np.argmax(off_pred,axis=-1)
    submission.to_csv(f"results/{file_name}/submission.csv",index=False)
    with open(f'./results/{file_name}/config.json','w') as f:
        json.dump(args.__dict__,f)
    with open(f'./results/{file_name}/logits.pkl','wb') as f:
        pickle.dump(off_pred,f)

    print(f"saved result in results/{file_name}/submission.csv")
    print("Test Inference is done !!")
    if config['wandb']:
        wandb.finish()

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    parser.add_argument('--seed', type=int, default=1010, help='random seed (default: 1010)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--dataset', type=str, default='MaskDataSet', help='dataset augmentation type (default: MaskDataSet)')
    parser.add_argument('--augmentation', type=str, default='MaskAugmentation', help='data augmentation type (default: MaskAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 256], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--kfold_splits', type=int, default=5, help='input k-fold splits number for cross validation (default: 5)')
    parser.add_argument('--model', type=str, default='MultiDropoutEfficientLite0', help='model type (default: MultiDropoutEfficientLite0)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type (default: AdamW)')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (default: 1e-2)')
    #parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='FocalLoss', help='criterion type (default: FocalLoss)')
    parser.add_argument('--weight_decay', type=int, default=1e-4, help='Weight decay for optimizer (default: 1e-4)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--lr_scheduler', type=str, default='OneCycleLR', help='learning scheduler (default: OneCycleLR)')
    parser.add_argument('--log_interval', type=int, default=30, help='how many batches to wait before logging training status')
    parser.add_argument('--file_name', default='exp', help='model save at {results}/{file_name}')
    parser.add_argument('--train_csv', default='/opt/ml/input/data/train/new_standard.csv', help='train data saved csv')
    parser.add_argument('--test_dir', default="/opt/ml/input/data/eval", help='test data saved directory')
    parser.add_argument('--mix_up', type=bool, default=False, help='if True, mix-up & cut-mix use')
    parser.add_argument('--num_class',type=int,default=18,help='input the number of class')
    parser.add_argument('--pseudo_label',type=bool,default=False,help='pseudo label usage')
    parser.add_argument('--pseudo_csv',type=str,default='/opt/ml/input/data/train/pseudo.csv',help='pseudo label usage')
    parser.add_argument('--wandb',type=bool,default=True,help='logging in WandB')
    parser.add_argument('--patience',type=int,default=5,help='early stopping patience number')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()

    with open(args.config,'r') as f:
        c = json.load(f)

    main(c)



