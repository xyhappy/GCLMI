import os, sys
os.chdir(sys.path[0])
import os.path as osp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from get_datasests import get_dataset
from torch_geometric.loader import DataLoader
from model import Encoder
from evaluate_embedding import evaluate_embedding
from arguments import arg_parse
from copy import deepcopy


class Net(nn.Module):
    def __init__(self, num_features, hidden_dim, num_gc_layers, flag=1):
        super(Net, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers, flag)  # GIN

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, device):

        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch, device)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug): 

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    args = arg_parse()
    if torch.cuda.is_available():
        if args.dataname == 'NCI1':
            args.device = 'cuda:0'
        elif args.dataname == 'PROTEINS':
            args.device = 'cuda:1'
        elif args.dataname == 'DD':
            args.device = 'cuda:1'
        elif args.dataname == 'MUTAG':
            args.device = 'cuda:0'
        elif args.dataname == 'COLLAB':
            args.device = 'cuda:0'
        elif args.dataname == 'REDDIT-BINARY':
            args.device = 'cuda:1'
        elif args.dataname == 'REDDIT-MULTI-5K':
            args.device = 'cuda:1'
        elif args.dataname == 'IMDB-BINARY':
            args.device = 'cuda:0'
    else:
        args.device = 'cpu'

    args.checkpoints = './{}/{}/{}'.format(args.model_path, args.dataname, args.experiment_number)
    if not osp.isdir(args.checkpoints):
        os.makedirs(args.checkpoints)
    
    final_acc = []
    for seed in [0, 1, 2, 3, 4]:
        args.seed = seed
        setup_seed(args.seed)
        accuracies = {'val': [], 'test': []}
        path = osp.join('./pyg_data', args.dataname)

        dataset = get_dataset(path, name=args.dataname).shuffle()
        dataset_eval = get_dataset(path, name=args.dataname).shuffle()
        args.num_features = dataset.num_features

        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
        model = Net(args.num_features, args.hidden_dim, args.num_gc_layers).to(args.device)
        model_aug = Net(args.num_features, args.hidden_dim, args.num_gc_layers, 0).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
       
        print('================')
        print('The graph number of {}: {}'.format(args.dataname, len(dataset)))
        print('lr: {}'.format(args.lr))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')

        model.eval()
        emb, y = model.encoder.get_embeddings(dataloader_eval, args.device)
        acc_val, acc = evaluate_embedding(emb, y)
        accuracies['val'].append(acc_val)
        accuracies['test'].append(acc)
        print('For seed {}, Before training, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
            args.seed, acc_val * 100, acc * 100))

        for epoch in range(1, args.epochs + 1):
            loss_all = 0
            model.train()
            for data in dataloader:
                data_aug = deepcopy(data)
                data = data.to(args.device)
                data_aug = data_aug.to(args.device)
                data_aug.x += F.normalize(torch.normal(0, torch.ones_like(data_aug.x) * args.sigma), dim=1)
                
                optimizer.zero_grad()

                x = model(data.x, data.edge_index, data.batch, args.device)
                x_aug1 = model(data_aug.x, data_aug.edge_index, data_aug.batch, args.device)
                x_aug2 = model_aug(data.x, data.edge_index, data.batch, args.device)

                loss1 = model.loss_cal(x, x_aug1)
                loss2 = model.loss_cal(x, x_aug2)
                loss3 = model_aug.loss_cal(x_aug1, x_aug2)
                loss = args.alpha * loss1 + (1-args.alpha) * loss2 - args.beta * loss3
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()

            print('Epoch: {}, Contrastive Loss: {:.4f}'.format(epoch, loss_all / len(dataloader)))

            if epoch % args.epochs == 0:
                torch.save(
                    model.state_dict(),
                    '{}/{}_seed={}_alpha={}_model.pth'.format(args.checkpoints, args.dataname,args.seed, args.alpha)
                )

            if epoch % args.log_interval == 0:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader_eval, args.device)
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                print('Epoch: {}, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
                    epoch, accuracies['val'][-1] * 100, accuracies['test'][-1] * 100))

        # model.load_state_dict(torch.load('{}/seed={}_model.pth'.format(
        #     args.checkpoints, args.seed), map_location=args.device))
        # model.eval()
        # emb, y = model.encoder.get_embeddings(dataloader_eval, args.device)
        # acc_val1, acc1 = evaluate_embedding(emb, y)
        # acc_val, acc = evaluate_embedding(emb, y)
        # accuracies['val'].append(acc_val)
        # accuracies['test'].append(acc)
        # print('For load model, For Seed {}, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
        #     args.seed, accuracies['val'][-1] * 100, accuracies['test'][-1] * 100))

        final_acc.append(accuracies['test'][-1])
    
    mean_acc = np.mean(final_acc) * 100
    mean_std = np.std(final_acc) * 100

    print('Final Test Acc: {:.2f} ± {:.2f}'.format(mean_acc, mean_std))



    
