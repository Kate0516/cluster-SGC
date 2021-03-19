import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from clustersgc.utils import fetch_normalization, sparse_mx_to_torch_sparse_tensor, sgc_precompute


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.每个subgraph进行了feature和adj的normalize
    """
    def __init__(self, model,cluster_method, cluster_num, test_ratio, graph, features, target):
        """
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray). [n,feature_num]
        :param target: Target vector (ndarray). [n,1]
        """
        self.graph = graph
        self.features = features
        self.target = target
        self.cluster_num = cluster_num
        self.cluster_method = cluster_method
        self.test_ratio = test_ratio
        self.model_name = model
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1]
        self.class_count = (np.max(self.target)+1).astype(int)#这里返回的是一个numpy64.float，会报错,因为Reddit的label是float

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.cluster_method == "metis":
            print("Metis graph clustering started.")
            self.metis_clustering()
        else:
            if self.cluster_num == 1:
                print('not use clustering')
            else:
                print("Random graph clustering started.")
            self.random_clustering()
        self.general_data_partitioning()
        #self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.cluster_num)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.cluster_num)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.sg_adjs = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            #print("fuck1{}".format(subgraph.nodes))
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.test_ratio)
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]
            self.sg_adjs[cluster] = nx.to_scipy_sparse_matrix(subgraph)
            self.sg_adjs[cluster] = self.sg_adjs[cluster] + self.sg_adjs[cluster].T
            adj_normalizer = fetch_normalization("AugNormAdj")
            self.sg_adjs[cluster] = adj_normalizer(self.sg_adjs[cluster])
            # if self.model_name == "SGC":
            #     adj = self.sg_adjs[cluster]
            #     adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #     adj = adj.to(device)
            #     features = torch.FloatTensor(self.sg_features[cluster])
            #     features = features.to(device)
            #     features, precompute_time = sgc_precompute(features, adj, self.degree)
            #     #precompute_time_all += precompute_time
            #     if torch.cuda.is_available():
            #         features = features.cpu()
            #     self.sg_features[cluster] = features.numpy()

    def precompute(self,degree):
        precompute_time_all = 0
        for cluster in self.clusters:
            adj = self.sg_adjs[cluster]
            adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adj = adj.to(device)
            features = torch.FloatTensor(self.sg_features[cluster])
            features = features.to(device)
            features, precompute_time = sgc_precompute(features, adj, degree)
            precompute_time_all += precompute_time
            if torch.cuda.is_available():
                features = features.cpu()
                self.sg_features[cluster] = features.numpy()
        return precompute_time_all

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            #这里应该封装一下
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_adjs[cluster] = sparse_mx_to_torch_sparse_tensor(self.sg_adjs[cluster]).float()
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            #print("fuck2{}".format(self.sg_train_nodes[cluster].shape))
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            #print("fuck3{}".format(self.sg_test_nodes[cluster].shape))
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            #self.sg_features[cluster] = (self.sg_features[cluster] - self.sg_features[cluster].mean(dim=0)) / self.sg_features[cluster].std(dim=0)
            #为什么这里会使metis失灵呢 SGC里citation的feature是稀疏矩阵，上面这句是Reddit不是稀疏矩阵的norm，clusterGCN没有normlizefeature
            #adj是SGC才会用到，edge是GCN才会用到
            #self.sg_targets[cluster] = torch.FloatTensor(self.sg_targets[cluster]) #for reddit
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster]) #我怕报错，还是改成了整型
