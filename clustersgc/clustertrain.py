import torch
import random
import numpy as np
from tqdm import trange, tqdm
from clustersgc.models import StackedGCN,SGC,GCN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import torch.nn.functional as F
from time import perf_counter

from clustersgc.utils import sgc_precompute
import matplotlib.pyplot as plt


class ClusterTrainer(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, model, layers, dropout, hidden,degree, epochs, lr, weight_decay, clustering_machine, plot=False):
        """
        :param clustering_machine:
        """
        self.model_name = model
        self.epochs = epochs
        self.layers = layers
        self.dropout = dropout
        self.degree = degree
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.plot = plot
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        if self.model_name == "SGC":
            self.model = SGC(self.clustering_machine.feature_count, self.clustering_machine.class_count)
        elif self.model_name == "SGCN":
            self.model = StackedGCN(self.layers, self.dropout, self.clustering_machine.feature_count,
                                    self.clustering_machine.class_count)
        elif self.model_name == "GCN":
            self.model = GCN(nfeat=self.clustering_machine.feature_count,
                    nhid=self.hidden,
                    nclass=self.clustering_machine.class_count,
                    dropout=self.dropout)
        self.model = self.model.to(self.device)

    def precompute(self):
        precompute_time_all = 0
        for cluster in self.clustering_machine.clusters:
            adj = self.clustering_machine.sg_adjs[cluster].to(self.device)
            features = self.clustering_machine.sg_features[cluster].to(self.device)
            features, precompute_time = sgc_precompute(features, adj, self.degree)
            precompute_time_all += precompute_time
        return precompute_time_all

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        adjs = self.clustering_machine.sg_adjs[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        #应该把这个搬到读取里
        #precompute_time = 0
        processed_features = features
        predictions = 0 #?
        if self.model_name == "SGC":
            #processed_features, precompute_time = sgc_precompute(features, adjs, self.degree)
            predictions = self.model(processed_features)
        elif self.model_name == "SGCN":
            predictions = self.model(edges,processed_features)
        elif self.model_name == "GCN":
            predictions = self.model(x=processed_features,adj=adjs)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        #average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        average_loss = F.cross_entropy(predictions[train_nodes], target[train_nodes])
        #print(average_loss)
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        self.accumulated_training_loss = self.accumulated_training_loss/self.node_count_seen

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        adjs = self.clustering_machine.sg_adjs[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        precompute_time = 0
        processed_features = features
        predictions = 0
        if self.model_name == "SGC":
            # processed_features, precompute_time = sgc_precompute(features, adjs, self.degree)
            predictions = self.model(processed_features)
        elif self.model_name == "SGCN":
            predictions = self.model(edges, processed_features)
        elif self.model_name == "GCN":
            predictions = self.model(processed_features, adjs)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        predictions = predictions[test_nodes,:]
        return predictions, target

    def train(self):
        """
        Training a model.
        """
        epochs = trange(self.epochs)
        #self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        #LBFGS必须传入closure，会导致再调用forward，套了一下，closure会报错
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()
        t = perf_counter()
        #precompute_time_all = 0
        xs = [0, 0]
        ys = [1, 1]
        if self.plot == True:
            plt.axis([0, 200, 0, 1])
            plt.ion()
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                #print("batch_average_loss: {}".format(batch_average_loss))
                batch_average_loss.backward()
                # def closure():
                #     self.optimizer.zero_grad()
                #     batch_average_loss, node_count = self.do_forward_pass(cluster)
                #     batch_average_loss.backward()
                #     return batch_average_loss, node_count
                # batch_average_loss, node_count = self.optimizer.step(closure)
                self.optimizer.step()
                self.update_average_loss(batch_average_loss, node_count)
                #precompute_time_all += precompute_time
                #print("epoch:{},loss up to %g",epoch,average_loss)
            # epochs.set_description("Train Loss: %g" % round(average_loss,4))
            if self.plot == True:
                xs[0] = xs[1]
                ys[0] = ys[1]
                xs[1] = epoch
                ys[1] = self.accumulated_training_loss
                plt.plot(xs, ys)
                plt.pause(0.1)
        train_time = perf_counter() - t
        return self.model, train_time

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        f1 = f1_score(self.targets, self.predictions, average="micro")
        ac = accuracy_score(self.targets, self.predictions)
        return f1, ac
