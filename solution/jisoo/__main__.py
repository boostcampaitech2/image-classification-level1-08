import train
import json
import argparse

def run(config):
    train.train(config)
    print(config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    parser.add_argument('--data_dir', type=str, default="./data/", help='path')
    parser.add_argument('--data_folder', type=str, default="./data/", help='path')
    parser.add_argument('--image_size', type=int, default= 512, help='image size')
    parser.add_argument('--enet_type', type=str, default='tf_efficientnet_b3_ns', help=' model type')
    parser.add_argument("--metric_strategy", type=bool, default=False, help='metruc strategy, default false')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, help='input number of workers')
    parser.add_argument('--init_lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--out_dim', type=int, default=3, help=' class num')
    parser.add_argument('--n_epochs', type=int, default=4, help='epoch')
    #parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--drop_nums', type=int, default=1, help='')
    parser.add_argument('--loss_type', type=str, default='focal_loss', help='criterion type (default: FocalLoss)')
    parser.add_argument('--use_amp', type=bool, default= False, help='use amplifier: false')
    parser.add_argument('--mixup_cutmix', type=bool, default= False, help='use mixup cutmix aug: false')
    parser.add_argument('--model_dir', type=str, default='./tf_efficientnet_b3_ns/weight_gender', help='ckpt model file')
    parser.add_argument('--log_dir', type=str, default='./tf_efficientnet_b3_ns/log_gender', help=' log save file')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str,default='0', help='device num')
    parser.add_argument('--fold', type=str, default="0,1,2,3,4", help='fold list in string')
    parser.add_argument('--pretrained', type=bool, default=True, help='if True,pretrained used')
    parser.add_argument('--eval',type=str,default='best',help='which ckpy to choose, best or final')
    parser.add_argument('--oof_dir',type=str,default='./tf_efficientnet_b3_ns/oofs/',help='out of fole test save dir')
    parser.add_argument('--auc_index',type=int,default=0,help='auc_index')
    # parser.add_argument('--wandb',type=bool,default=True,help='logging in WandB')
    # parser.add_argument('--patience',type=int,default=5,help='early stopping patience number')
    # Container environment

    args = parser.parse_args()
    with open(args.config,'r') as f:
        c = json.load(f)
    c.update(args.__dict__)
    run(c)
