from util.load_data import load_data_set
from XSimGCL import XSimGCL
from util.conf import args
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_set = load_data_set(args.training_set)
test_set = load_data_set(args.test_set)
model = XSimGCL(args, train_set, test_set).to(device)
model.train()
