# 问题1 clustering中的featurenorm为什么会使准确率下降（metis严重下降）
#本实验中目前没有对feature进行norm,对adj进行了norm，SGC中对Reddit和citation分别进行了不同的feature和adj norm，clusterGCN没有norm
# 问题2 randomseed为什么不能使结果不变，setseedv1似乎能解决问题
import torch
import scipy.sparse as sp
import numpy as np
import networkx as nx
from clustersgc.clustering import ClusteringMachine
from clustersgc.clustertrain import ClusterTrainer
from clustersgc.utils import load_reddit,load_npz,set_seed
import pickle as pkl

SEED=42

DATASET_DIR="datasets/"
DATASET_NAME = "pubmed"
TEST_RATIO=0.1 #0.9 for clustersgc

CLUSTER =True
CLUSTER_METHOD="metis" #metis random
CLUSTER_NUM=3

MODEL="SGC"
#SGC
DEGREE=1 #2 only when sgc
#GCN&SGCN
LAYERS = [16,16,16] ##SGCN,defalut16,16,16
HIDDEN = 317 #GCN
DROPOUT=0.5 #defalut 0.5

EPOCHS=200 #200 for clustergcn 100 for sgc
LR=0.2 #0.01 for clustergcn 0.2 for sgc
WEIGHT_DECAY=0 #5e-6 for sgc

TUNED = True
if TUNED:
        with open("cluster-tuning/{}-lr.txt".format(DATASET_NAME), 'rb') as f:
            opt = pkl.load(f)
            #WEIGHT_DECAY =opt['weight_decay']
            LR = opt['lr']
            #DEGREE = int(opt['degree'])
            #print("using tuned weight decay: {} lr:{} degree:{}".format(WEIGHT_DECAY,LR,DEGREE))
            print("using tuned lr: {}".format(LR))
        with open("cluster-tuning/{}-weight_decay.txt".format(DATASET_NAME), 'rb') as f2:
            opt = pkl.load(f2)
            #WEIGHT_DECAY =opt['weight_decay']
            WEIGHT_DECAY = opt['weight_decay']
            #DEGREE = int(opt['degree'])
            #print("using tuned weight decay: {} lr:{} degree:{}".format(WEIGHT_DECAY,LR,DEGREE))
            print("using tuned weight_decay: {}".format(WEIGHT_DECAY))

set_seed(SEED)
# np.random.seed(SEED)
# if torch.cuda.is_available():
#     print("using cuda")
#     torch.cuda.manual_seed(SEED)
# else:
#     print("using cpu")
#     torch.manual_seed(SEED)

if DATASET_NAME == "reddit":
    graph, features, labels = load_reddit(DATASET_DIR)
else:
    graph, features, labels= load_npz(DATASET_DIR,DATASET_NAME)
print("Finished data loading.")

if CLUSTER == False:
    CLUSTER_METHOD = "random"
    CLUSTER_NUM = 1

clustering_machine = ClusteringMachine(MODEL,CLUSTER_METHOD,CLUSTER_NUM,TEST_RATIO,graph, features, labels)#normalize
clustering_machine.decompose()
print("Finished decompose")

precompute_time = 0
if MODEL == "SGC":
    precompute_time = clustering_machine.precompute(DEGREE)

clustering_machine.transfer_edges_and_nodes()

gcn_trainer = ClusterTrainer(MODEL, LAYERS, DROPOUT, HIDDEN,DEGREE, EPOCHS,LR,WEIGHT_DECAY,clustering_machine,plot=False)
model, train_time = gcn_trainer.train()
#SGC里用的LBGFS优化，这里面因为代码结构用的Adam

test_f1,test_ac = gcn_trainer.test() #gcn doesn't precompute
print("Train Time: {:.4f}s, Precompute Time: {:.4f}s, F1: {:.4f} AC:{:.4f}".format(train_time,precompute_time,test_f1,test_ac))