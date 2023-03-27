import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--training_set', default="./dataset/ml-1M/train.txt", type=str, help='train_data')
    parser.add_argument('--test_set', default="./dataset/ml-1M/test.txt", type=str, help='test_data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch', default=1024, type=int, help='batch size')
    parser.add_argument('--cl_rate', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.25, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.15, type=float, help='temperature in cl loss')
    parser.add_argument('--reg_lambda', default=1e-4, type=float, help='l2 reg weight')
    parser.add_argument('--eps', default=0.2, type=float)
    parser.add_argument('--l', default=1, type=int)
    parser.add_argument('--emb_size', default=64, type=int)
    return parser.parse_args()


args = parse_args()
