import train
import json
import argparse

def run(config):
    train.train(config)

if __name__=='__main__':
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

    args = parser.parse_args()
    with open(args.config,'r') as f:
        c = json.load(f)
    c.update(args.__dict__)
    run(c)
