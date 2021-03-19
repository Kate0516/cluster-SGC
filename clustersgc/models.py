#修改了原SGC的forward，增加了relu
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        features = self.W(x)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)
    #GCN因为有多层，要一直保证系数矩阵的normal
    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class StackedGCN(torch.nn.Module):
    """
    Multi-layer GCN model.
    """
    def __init__(self, layers, dropout, input_channels, output_channels):
        """
        :input_channels: Number of features.
        :output_channels: Number of target features.
        """
        super(StackedGCN, self).__init__()
        self.layers_arg = layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.setup_layers()

    def setup_layers(self):
        self.layers = []
        self.layers_arg = [self.input_channels] + self.layers_arg + [self.output_channels]
        for i, _ in enumerate(self.layers_arg[:-1]):
            self.layers.append(GCNConv(self.layers_arg[i],self.layers_arg[i+1]))
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):
        """
        Making a forward pass.
        :param edges: Edge list LongTensor.
        :param features: Feature matrix input FLoatTensor.
        :return predictions: Prediction matrix output FLoatTensor.
        """
        for i, _ in enumerate(self.layers_arg[:-2]):
            features = torch.nn.functional.relu(self.layers[i](features, edges))
            if i>1:
                features = torch.nn.functional.dropout(features, p = self.dropout, training = self.training)
        features = self.layers[i+1](features, edges)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *layers):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in layers:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

# def get_model(model_opt, nfeat, nclass, layers, dropout, device):
#     if model_opt == "GCN":
#         model = StackedGCN(layers, dropout, nfeat, nclass)
#     elif model_opt == "SGC":
#         model = SGC(nfeat=nfeat,
#                     nclass=nclass)
#     else:
#         raise NotImplementedError('model:{} is not implemented!'.format(model_opt))
#     return model