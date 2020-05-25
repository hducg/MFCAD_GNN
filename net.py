# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:38:16 2019

@author: 2624224
"""
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import EdgeConv

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
class Net(torch.nn.Module):
    def __init__(self, out_channels, aggr='sum'):
        super(Net, self).__init__()

        self.conv1 = EdgeConv(MLP([2 * 4, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 64, 64]), aggr)
        self.conv3 = EdgeConv(MLP([2 * 64, 64, 64]), aggr)
        self.lin1 = MLP([3 * 64, 1024])

        self.mlp = Seq(
            MLP([1024, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5),
            Lin(128, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)
    
    