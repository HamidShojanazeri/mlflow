from __future__ import print_function
import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from fastprogress.fastprogress import master_bar, progress_bar
import argparse
import mlflow
import mlflow.pytorch
from mlflow.pytorch.callbacks import *
import tempfile
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter




parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--enable-cuda', type=str, choices=['True', 'False'], default='Flase',
                    help='enables or disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

enable_cuda_flag = True if args.enable_cuda == 'True' else False

args.cuda = enable_cuda_flag and torch.cuda.is_available()
print("********************************* CUDA is set to:",args.cuda)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

seed_torch(args.seed)
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = F.nll_loss

output_dir = dirpath = tempfile.mkdtemp()
writer = SummaryWriter(output_dir)

class MLFlowTracking_log_artifacts(Callback):
    def __init__(self):
        self.train_stats, self.valid_stats = AverageMetrics([accuracy], True), AverageMetrics([accuracy], False)
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)
        for item in self.train_stats.avg_stats:
            writer.add_scalar("train_loss", self.train_stats.avg_stats[0],self.epoch)
            writer.add_scalar("train_accuracy", self.train_stats.avg_stats[1].item(),self.epoch)
        for item in self.valid_stats.avg_stats:
            writer.add_scalar("valid_loss", self.valid_stats.avg_stats[0],self.epoch)
            writer.add_scalar("valid_accuracy", self.valid_stats.avg_stats[1].item(),self.epoch)
        mlflow.log_artifacts(output_dir, artifact_path="events")


cbfs = [MLFlowTracking_log_artifacts]
learn = get_learner(model,optimizer, loss_fn,train_loader, test_loader)
run = autolog(cb_funcs=cbfs)
run.fit(2, learn)
