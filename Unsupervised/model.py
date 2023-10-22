import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, PReLU
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, flag=1):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.flag = flag
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if self.flag:   #GIN
                if i:
                    nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                else:
                    nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                conv = GINConv(nn)
            else:   #MLP
                if i:
                    conv = Sequential(Linear(dim, dim), ReLU())
                else:
                    conv = Sequential(Linear(num_features, dim), ReLU())
                    
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)
        self.projection = Sequential(Linear(dim*num_gc_layers, dim*num_gc_layers), ReLU(inplace=True), Linear(dim*num_gc_layers, dim*num_gc_layers))

    def forward(self, x, edge_index, batch, mark=1):
        xs = []
        for i in range(self.num_gc_layers):
            if self.flag:   #GIN
                x = F.relu(self.convs[i](x, edge_index))
            else:   #MLP
                x = F.relu(self.convs[i](x))
            x = self.bns[i](x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, dim=1)
        if mark == 1:
            x = self.projection(x)
        return x

    def get_embeddings(self, loader, device):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x = self.forward(x, edge_index, batch, 0) 

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
    

class Net(torch.nn.Module):
    
    def __init__(self, num_features, hidden_dim, num_gc_layers, alpha, beta, sigma):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.embedding_dim = hidden_dim * num_gc_layers
        self.main_encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.pretext_encoder = Encoder(num_features, hidden_dim, num_gc_layers, 0)

    def loss_infomax(self, x, x_cl): 
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)  #|x|
        x_cl_abs = x_cl.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss
    
    def loss_infomin(self, x, x_cl):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)  #|x|
        x_cl_abs = x_cl.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_cl) / torch.einsum('i,j->ij', x_abs, x_cl_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = torch.log(loss+1).mean()
        return loss
     
    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        x_aug = x + F.normalize(torch.normal(0, torch.ones_like(x)*self.sigma), dim=1)

        gx = self.main_encoder(x, edge_index, batch)
        gx1 = self.main_encoder(x_aug, edge_index, batch)
        gx2 = self.pretext_encoder(x, edge_index, batch)
    
        loss1 = self.loss_infomax(gx, gx1)
        loss2 = self.loss_infomax(gx, gx2)
        loss3 = self.loss_infomin(gx1, gx2)

        loss = self.alpha*loss1 + (1-self.alpha)*loss2 + self.beta*loss3
                       
        return loss
    
