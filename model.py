import torch
from torch import nn


class MLP(torch.nn.Module):
    # adopt a MLP as classifier for graphs
    def __init__(self,input_size):
        super(MLP, self).__init__()
        self.nn = nn.BatchNorm1d(input_size)
        self.linear1 = torch.nn.Linear(input_size,input_size*20)
        self.linear2 = torch.nn.Linear(input_size*20,input_size*20)
        self.linear3 = torch.nn.Linear(input_size*20,input_size*10)
        self.linear4 = torch.nn.Linear(input_size*10,input_size)
        self.linear5 = torch.nn.Linear(input_size,1)
        self.act= nn.ReLU()
    def forward(self, x):
        out = self.nn(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        out = self.linear3(out)
        out = self.act(out)
        out = self.linear4(out)
        out = self.act(out)
        out = nn.functional.dropout(out, p=0.5, training=self.training)
        out = self.linear5(out)
        return out