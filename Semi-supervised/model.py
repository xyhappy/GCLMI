import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv


class Encoder(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, num_gc_layers, residual, num_fc_layers, global_pool, dropout, flag=1):
        super(Encoder, self).__init__()
        self.conv_residual = residual
        self.global_pool = global_pool
        self.dropout = dropout
        self.flag = flag
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            bn = torch.nn.BatchNorm1d(hidden) if i else torch.nn.BatchNorm1d(num_features)
            self.bns_conv.append(bn)

        for i in range(num_gc_layers):
            if self.flag: #GCN
                conv = GCNConv(hidden, hidden) if i else GCNConv(num_features, hidden)
            else: #MLP
                conv = Sequential(Linear(hidden, hidden), ReLU()) if i else Sequential(Linear(num_features, hidden), ReLU())
            self.convs.append(conv)

        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.lin_class = Linear(hidden, num_classes)

        self.proj_head = Sequential(Linear(hidden, hidden), ReLU(inplace=True), Linear(hidden, hidden))
            
    
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
    
    
    def forward_cl(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            if self.flag: #GCN
                x_ = F.relu(conv(x_, edge_index))
            else: #MLP
                x_ = F.relu(conv(x_))
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x
    

class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, num_gc_layers,
                 num_fc_layers, residual, global_pool, dropout, alpha, beta, sigma):
        super(Net, self).__init__()
        self.conv_residual = residual 
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.main_encoder = Encoder(num_features, num_classes, hidden, num_gc_layers, residual, num_fc_layers, self.global_pool, dropout)
        self.pretext_encoder = Encoder(num_features, num_classes, hidden, num_gc_layers, residual, num_gc_layers, self.global_pool, dropout, 0)
        
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)


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
    
    
    def forward_cl(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        x_aug = x + F.normalize(torch.normal(0, torch.ones_like(x)*self.sigma), dim=1)

        gx = self.main_encoder.forward_cl(x, edge_index, batch)
        gx1 = self.main_encoder.forward_cl(x_aug, edge_index, batch)
        gx2 = self.pretext_encoder.forward_cl(x, edge_index, batch)

        loss1 = self.loss_infomax(gx, gx1)
        loss2 = self.loss_infomax(gx, gx2)
        loss3 = self.loss_infomin(gx1, gx2)

        loss = self.alpha*loss1 + (1-self.alpha)*loss2 + self.beta*loss3

        return loss
    

    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        xg = self.main_encoder(x, edge_index, batch)
        return xg


    def __repr__(self):
        return self.__class__.__name__
