import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from time import perf_counter

import random

# def set_seed(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

# def row_normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#
# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

# def set_seed(seed, cuda):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if cuda: torch.cuda.manual_seed(seed)

def load_reddit(dataset_dir):
    #没有normalize也没有转换成tensor也没有cuda
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")
    graph = nx.from_scipy_sparse_matrix(adj)
    features = data['feats']
    labels = np.zeros(adj.shape[0])
    labels[data['train_index']]  = data['y_train']
    labels[data['val_index']]  = data['y_val']
    labels[data['test_index']]  = data['y_test']
    labels = labels.reshape(-1, 1) #方便cluster
    return graph, features, labels

def load_npz(dataset_dir, dataset_name):
    #载入graph gallery中的npz数据集
    dataset_name = dataset_name + ".npz"
    data = np.load(dataset_dir + dataset_name,allow_pickle=True) #['adj_matrix', 'node_attr', 'node_label'] 不加allowpickle会报错
    adj = data['adj_matrix']
    adj = adj.all() #adj是0维的ndarray
    graph = nx.from_scipy_sparse_matrix(adj)
    features = data['node_attr']
    features = features.all()
    features = np.array(features.todense())
    labels = data['node_label']
    labels = labels.reshape(-1, 1)
    return graph, features, labels