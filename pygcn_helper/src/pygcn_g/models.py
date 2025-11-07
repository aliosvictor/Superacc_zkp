from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer as introduced by Kipf & Welling (2017).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.size(1))
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input_tensor, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network tailored for the Cora benchmark.
    """

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.gc1(features, adjacency))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        logits = self.gc2(hidden, adjacency)
        return F.log_softmax(logits, dim=1)
