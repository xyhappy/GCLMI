import sys

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 10 == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.4f}, Test Accuracy: {:.4f}'.format(
            fold+1, epoch, train_acc, test_acc))
    sys.stdout.flush()


