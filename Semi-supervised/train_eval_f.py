import sys
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from model import Net


def cross_validation_with_val_set_f(dataset, args, logger):
    assert args.epoch_select in ['val_max', 'test_max'], args.epoch_select

    val_losses, train_accs, test_accs = [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, args.folds, args.epoch_select, args.semi_split))):
        
        #input data
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        #model
        model = Net(args.num_features, args.num_classes, args.hidden, args.num_gc_layers,
                 args.num_fc_layers, args.skip_connection, args.global_pool, args.dropout, args.alpha, args.beta, args.sigma).to(args.device)
        model.load_state_dict(torch.load(args.model_PATH, map_location=args.device))
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        #iteration
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, optimizer, train_loader, args.device)
            train_accs.append(train_acc)
            val_losses.append(eval_loss(model, val_loader, args.device, args.with_eval_mode))
            test_accs.append(eval_acc(model, test_loader, args.device, args.with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }
            if logger is not None:
                logger(eval_info)

    #accuracy
    train_acc, test_acc = torch.tensor(train_accs), torch.tensor(test_accs)
    val_loss = torch.tensor(val_losses)
    train_acc = train_acc.view(args.folds, args.epochs) # [10, 100]
    test_acc = test_acc.view(args.folds, args.epochs)
    val_loss = val_loss.view(args.folds, args.epochs)
    if args.epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0) 
        selected_epoch = selected_epoch.repeat(args.folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch] 
    
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()

    print('For seed={}, model_epoch={}, Train Acc: {:.2f}, Test Acc: {:.2f} ± {:.2f}'.
          format(args.seed, args.model_epoch, train_acc_mean*100, test_acc_mean*100, test_acc_std*100))
    sys.stdout.flush()

    return test_acc_mean


def k_fold(dataset, folds, epoch_select, semi_split):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)] 
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(semi_split, shuffle=True, random_state=12345) 
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
            idx_train = idx_train[idx]
            break

        train_indices.append(idx_train) 

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, device)
        loss = F.nll_loss(out, data.y.view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch, device).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch, device)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
