import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch.nn import Sequential, BatchNorm1d, Linear, PReLU


class ResMLP(torch.nn.Module):
    """MLP with BN and residual connection."""
    def __init__(self, dataset, hidden, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0):
        super(ResMLP, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.
        self.res_branch = res_branch  # "BNConvReLU"
        self.collapse = collapse  #False
        assert "sum" in global_pool or "mean" in global_pool, global_pool
        if "sum" in global_pool:  # Yes
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool
        self.dropout = dropout

        if "xg" in dataset[0]:  # Utilize graph level features.  # False
            self.use_xg = True
            self.bn1_xg = BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = BatchNorm1d(hidden)
            self.lin2_xg = Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features
        if collapse:  # False
            self.bn_feat = BatchNorm1d(hidden_in)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            if "gating" in global_pool:  # False
                self.gating = torch.nn.Sequential(
                    Linear(hidden_in, hidden_in),
                    torch.nn.ReLU(),
                    Linear(hidden_in, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden_in))
                self.lins.append(Linear(hidden_in, hidden))
                hidden_in = hidden
            self.lin_class = Linear(hidden_in, dataset.num_classes)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.conv_feat = MLP(hidden_in, hidden) 
            if "gating" in global_pool: # False
                self.gating = torch.nn.Sequential(
                    Linear(hidden, hidden),
                    torch.nn.ReLU(),
                    Linear(hidden, 1),
                    torch.nn.Sigmoid())
            else:
                self.gating = None
            self.bns_conv = torch.nn.ModuleList()
            self.convs = torch.nn.ModuleList()
            if self.res_branch == "resnet":  # False
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(MLP(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(MLP(hidden, hidden))
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(MLP(hidden, hidden))
            else:  # "BNConvReLU"
                for i in range(num_conv_layers):
                    self.bns_conv.append(BatchNorm1d(hidden))
                    self.convs.append(MLP(hidden, hidden))
            self.bn_hidden = BatchNorm1d(hidden)
            self.bns_fc = torch.nn.ModuleList()
            self.lins = torch.nn.ModuleList()
            for i in range(num_fc_layers - 1):
                self.bns_fc.append(BatchNorm1d(hidden))
                self.lins.append(Linear(hidden, hidden))
            self.lin_class = Linear(hidden, dataset.num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

        self.proj_head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 128))

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, batch = data.x, data.batch
        if self.use_xg:  # False
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        if self.collapse:
            return self.forward_collapse(x, batch, xg)
        elif self.res_branch == "BNConvReLU":  # Yes
            return self.forward_BNConvReLU(x, batch, xg)
        elif self.res_branch == "BNReLUConv":
            return self.forward_BNReLUConv(x, batch, xg)
        elif self.res_branch == "ConvReLUBN":
            return self.forward_ConvReLUBN(x, batch, xg)
        elif self.res_branch == "resnet":
            return self.forward_resnet(x, batch, xg)
        else:
            raise ValueError("Unknown res_branch %s" % self.res_branch)

    def forward_collapse(self, x, batch, xg=None):
        x = self.bn_feat(x)
        gate = 1 if self.gating is None else self.gating(x)
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    ### This one!!!
    def forward_BNConvReLU(self, x, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_))
            x = x + x_ if self.conv_residual else x_  # x_
        gate = 1 if self.gating is None else self.gating(x)  # 1
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg  # x
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_  # x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_cl(self, data):
        x, batch = data.x, data.batch
        if self.use_xg:  # False
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        return self.forward_BNConvReLU_cl(x, batch, xg)

    def forward_BNConvReLU_cl(self, x, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_))
            x = x + x_ if self.conv_residual else x_  # x_
        gate = 1 if self.gating is None else self.gating(x)  # 1
        x = self.global_pool(x * gate, batch)
        x = x if xg is None else x + xg  # x
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_  # x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.proj_head(x)
        return x

    def forward_BNReLUConv(self, x, batch, xg=None):
        x = self.bn_feat(x)
        x = self.conv_feat(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(self.bns_conv[i](x))
            x_ = conv(x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_ if self.fc_residual else x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def loss_cl(self, x1, x2):
        T = 0.5
        batch_size, _ = x1.size()
        
        # batch_size *= 2
        # x1, x2 = torch.cat((x1, x2), dim=0), torch.cat((x2, x1), dim=0)

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        '''
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        self_sim = sim_matrix[range(batch_size), list(range(int(batch_size/2), batch_size))+list(range(int(batch_size/2)))]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim - self_sim)
        loss = - torch.log(loss).mean()
        '''
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        
        return loss


    def forward_ConvReLUBN(self, x, batch, xg=None):
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x))
        x = self.bn_hidden(x)
        for i, conv in enumerate(self.convs):
            x_ = F.relu(conv(x))
            x_ = self.bns_conv[i](x_)
            x = x + x_ if self.conv_residual else x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(lin(x))
            x_ = self.bns_fc[i](x_)
            x = x + x_ if self.fc_residual else x_
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def forward_resnet(self, x, batch, xg=None):
        # this mimics resnet architecture in cv.
        x = self.bn_feat(x)
        x = self.conv_feat(x)
        for i in range(len(self.convs) // 3):
            x_ = x
            x_ = F.relu(self.bns_conv[i*3+0](x_))
            x_ = self.convs[i*3+0](x_)
            x_ = F.relu(self.bns_conv[i*3+1](x_))
            x_ = self.convs[i*3+1](x_)
            x_ = F.relu(self.bns_conv[i*3+2](x_))
            x_ = self.convs[i*3+2](x_)
            x = x + x_
        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = F.relu(self.bns_fc[i](x))
            x_ = lin(x_)
            x = x + x_
        x = F.relu(self.bn_hidden(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = Sequential(Linear(in_dim, out_dim), PReLU(),
                              Linear(out_dim, out_dim), PReLU(),
                              Linear(out_dim, out_dim), PReLU()
        )
        self.linear_shortcut = Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)