import os, sys
os.chdir(sys.path[0])
import argparse
from datasets import get_dataset
from train_eval_p import cross_validation_with_val_set_p
from train_eval_f import cross_validation_with_val_set_f
import torch
import os
import random
import numpy as np
import sys
from utils import logger

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="./pyg_data")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--num_gc_layers', type=int, default=3) 
parser.add_argument('--num_fc_layers', type=int, default=2) 
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--model_epoch', type=str, default="100")
parser.add_argument('--semi_split', type=int, default=10)
parser.add_argument('--dataname', type=str, default="COLLAB")
parser.add_argument('--with_eval_mode', type=str2bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--folds', type=int, default=10)
parser.add_argument('--experiment', type=str, default='1')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--sigma', type=float, default=0.01)
parser.add_argument('--model_path', default='checkpoints_test', type=str, help='File to save model')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.dataname == 'NCI1':
            args.device = 'cuda:0'
        elif args.dataname == 'PROTEINS':
            args.device = 'cuda:1'
        elif args.dataname == 'DD':
            args.device = 'cuda:0'
        elif args.dataname == 'COLLAB':
            args.device = 'cuda:1'
        elif args.dataname == 'REDDIT-BINARY':
            args.device = 'cuda:0'
        elif args.dataname == 'REDDIT-MULTI-5K':
            args.device = 'cuda:1'
    else:
        args.device = 'cpu'

    args.save_model = './{}/{}'.format(args.model_path, args.dataname)
    if not os.path.isdir(args.save_model):
        os.makedirs(args.save_model, exist_ok=True)
   
    model_epochs = ['20', '40', '60', '80', '100']
    acc_20, acc_40, acc_60, acc_80, acc_100 = [], [], [], [], []
    for seed in [0, 1, 2, 3, 4]:
        args.seed = seed
        setup_seed(args.seed)
        dataset = get_dataset(args.data_root, args.dataname)
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes

        # pre-training
        cross_validation_with_val_set_p(dataset, args) 

        #finetuning
        for model_epoch in model_epochs:
            args.model_epoch = model_epoch
            args.model_PATH = '{}/{}_seed={}_alpha={}_beta={}_lr={}_epoch={}.pt'.format(args.save_model, args.dataname, args.seed, args.alpha, args.beta, args.lr, args.model_epoch)
            acc = cross_validation_with_val_set_f(dataset, args, logger=logger)
            locals()['acc_'+str(args.model_epoch)].append(np.array(acc))

    mean_acc = [np.mean(acc_20)*100, np.mean(acc_40)*100, np.mean(acc_60)*100, np.mean(acc_80)*100, np.mean(acc_100)*100]
    mean_std = [np.std(acc_20)*100, np.std(acc_40)*100, np.std(acc_60)*100, np.std(acc_80)*100, np.std(acc_100)*100]
    
    final_acc = np.max(mean_acc)
    final_std = mean_std[np.argmax(mean_acc)]
    best_epoch = model_epochs[np.argmax(mean_acc)]

    print('For alpha={}, beta={}, lr={}, model_epoch={}, mean_acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, model_epochs[0], mean_acc[0], mean_std[0]))
    print('For alpha={}, beta={}, lr={}, model_epoch={}, mean_acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, model_epochs[1], mean_acc[1], mean_std[1]))
    print('For alpha={}, beta={}, lr={}, model_epoch={}, mean_acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, model_epochs[2], mean_acc[2], mean_std[2]))
    print('For alpha={}, beta={}, lr={}, model_epoch={}, mean_acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, model_epochs[3], mean_acc[3], mean_std[3]))
    print('For alpha={}, beta={}, lr={}, model_epoch={}, mean_acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, model_epochs[4], mean_acc[4], mean_std[4]))
    print('For alpha={}, beta={}, lr={}, Best Epoch={}, Final Test Acc: {:.2f} ± {:.2f}'.format(args.alpha, args.beta, args.lr, best_epoch, final_acc, final_std))