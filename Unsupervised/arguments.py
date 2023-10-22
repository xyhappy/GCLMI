import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Graph_Contrastive_learning')
    parser.add_argument('--dataname', dest='dataname', type=str, default='MUTAG', help='Dataname')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=3, help='Number of graph convolution layers')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--log_interval', type=int, default=10, help='log_interval')
    parser.add_argument('--experiment_number', default='1', type=str)
    parser.add_argument('--model_path', default='checkpoints', type=str, help='File to save model')
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--sigma', default=0.01, type=float)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()

