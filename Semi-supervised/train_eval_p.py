import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from model import Net
import time


def cross_validation_with_val_set_p(dataset, args):
    assert args.epoch_select in ['val_max', 'test_max'], args.epoch_select
    model = Net(args.num_features, args.num_classes, args.hidden, args.num_gc_layers,
                 args.num_fc_layers, args.skip_connection, args.global_pool, args.dropout, args.alpha, args.beta, args.sigma).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    # time begin
    # if torch.cuda.is_available():
    #         torch.cuda.synchronize(args.device)
    # t_start = time.perf_counter()

    dataset = dataset.shuffle()
    loader = DataLoader(dataset, args.batch_size, shuffle=False)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, optimizer, loader, args.device)
        print('For seed: {}, epoch: {}, pre-train loss: {:.2f}'.format(args.seed, epoch, train_loss))
        if epoch % 20 == 0:
            torch.save(model.state_dict(), '{}/{}_seed={}_alpha={}_beta={}_lr={}_epoch={}.pt'.format(args.save_model, args.dataname, args.seed, args.alpha, args.beta, args.lr, epoch))
            
    print("The pre-training of seed {} have finished!".format(args.seed))
    
    # time over
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize(args.device)
    # t_end = time.perf_counter()
    # print('Duration: {:.3f}'.format(t_end - t_start))

    
def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        loss = model.forward_cl(data.x, data.edge_index, data.batch, device)
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)






