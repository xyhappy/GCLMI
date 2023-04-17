import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch.autograd import Variable
from res_gcn import ResGCN
from res_mlp import ResMLP
from copy import deepcopy
import numpy as np
import time


def cross_validation_with_val_set_p(dataset, args):
    assert args.epoch_select in ['val_max', 'test_max'], args.epoch_select

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, args.folds, args.epoch_select))):

        model = ResGCN(dataset, args.hidden, args.n_layers_feat, args.n_layers_conv,
                            args.n_layers_fc, gfn=False, collapse=False,
                            residual=args.skip_connection, res_branch=args.res_branch,
                            global_pool=args.global_pool, dropout=args.dropout,
                            edge_norm=args.edge_norm).to(args.device)

        model_aug = ResMLP(dataset, args.hidden, args.n_layers_feat, args.n_layers_conv,
                            args.n_layers_fc, collapse=False,
                            residual=args.skip_connection, res_branch=args.res_branch,
                            global_pool=args.global_pool, dropout=args.dropout).to(args.device)

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0) 


        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)
        for epoch in range(1, args.epochs + 1):
            train_loss = train(
                model, model_aug, optimizer, dataset, args.device, args.batch_size, args.sigma, args.alpha, args.beta)

            print('For suffix: {}, epoch: {}, pre-train loss: {:.4f}'.format(args.suffix, epoch, train_loss))

            if epoch % args.lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr_decay_factor * param_group['lr']

            if epoch % 20 == 0:
                torch.save(model.state_dict(), '{}/{}_lr={}_alpha={}_beta={}_epoch={}_suffix={}_model.pt'.format(args.save_model, args.dataname, args.lr, args.alpha, args.beta, epoch, args.suffix))

        print("The pre-training of suffix {} have finished!".format(args.suffix))
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(args.device)

        break


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def train(model, model_aug, optimizer, dataset, device, batch_size, sigma, alpha, beta):

    dataset = dataset.shuffle()
    loader = DataLoader(dataset, batch_size, shuffle=False)
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data_aug = deepcopy(data)
        data = data.to(device)
        data_aug = data_aug.to(device)
        data_aug.x += F.normalize(torch.normal(0, torch.ones_like(data_aug.x) * sigma), dim=1)
        out = model.forward_cl(data)
        out1 = model.forward_cl(data_aug)
        out2 = model_aug.forward_cl(data)
        out2 = Variable(out2.detach().data, requires_grad=False)
        loss1 = model.loss_cl(out, out1)
        loss2 = model.loss_cl(out, out2)
        loss3 = model_aug.loss_cl(out1, out2)
        loss = alpha * loss1 + (1-alpha) * loss2 - beta * loss3
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)






