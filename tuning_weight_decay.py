import pickle as pkl
import os
from math import log
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt

from clustersgc.clustering import ClusteringMachine
from clustersgc.clustertrain import ClusterTrainer
from clustersgc.utils import load_reddit,load_npz,set_seed
import pickle as pkl

SEED=42

DATASET_DIR="datasets/"
DATASET_NAME = "cora_full"

CLUSTER =True
CLUSTER_METHOD="metis"
CLUSTER_NUM=3
TEST_RATIO=0.1 #0.9 for clustersgc
DEGREE=2 #2 only when sgc

MODEL="SGC"
LAYERS = [16,16,16] #16,16,16 only when gcn
DROPOUT=0.5 #0.5 only when gcn

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
set_seed(SEED)

# Hyperparameter optimization
space = {'weight_decay' : hp.loguniform('weight_decay', log(1e-10), log(1e-4)),
         #'lr' : hp.uniform('lr', 0.005, 0.2), #0.2for cora citerseer, 0.5for pubmed
         #'degree' : hp.quniform('degree', 1,5,4),
        #'test_ratio' : hp.uniform('test_ratio', 0.1,1)
         }

if DATASET_NAME == "reddit":
    graph, features, labels = load_reddit(DATASET_DIR)
else:
    graph, features, labels= load_npz(DATASET_DIR,DATASET_NAME)
print("Finished data loading.")

if CLUSTER == False:
    CLUSTER_METHOD = "random"
    CLUSTER_NUM = 1

clustering_machine = ClusteringMachine(MODEL, DEGREE, CLUSTER_METHOD,CLUSTER_NUM , TEST_RATIO, graph, features,
                                           labels)  # normalize
precompute_time = clustering_machine.decompose()
print("Finished decompose")

def clustersgc_objective(space):
    gcn_trainer = ClusterTrainer(MODEL, LAYERS, DROPOUT, DEGREE, EPOCHS,LR,space['weight_decay'],clustering_machine)
    model, train_time = gcn_trainer.train()
    test_f1,test_ac = gcn_trainer.test()
    # print('weight decay: {:.2e} lr:{}'.format(space['weight_decay'], space['lr']) + 'f1score: {:.4f}'.format(test_f1))
    print('weight_decay:{}'.format(space['weight_decay']) + 'f1score: {:.4f}'.format(test_f1))
    return {'loss': -test_f1, 'status': STATUS_OK}

trials = Trials()
best = fmin(clustersgc_objective, space=space, algo=tpe.suggest, max_evals=200,trials=trials)
#print("Best weight decay: {:.2e} lr:{}".format(best["weight_decay"],best['lr']))
print("Best weight_decay:{}".format(best['weight_decay']))

os.makedirs("./cluster-tuning", exist_ok=True)
path = 'cluster-tuning/{}-weight_decay.txt'.format(DATASET_NAME)
with open(path, 'wb') as f: pkl.dump(best, f)

clustersgc_objective, ax = plt.subplots(1)
xs = [t['misc']['vals']['weight_decay'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$weight_decay$', fontsize=16)
ax.set_ylabel('$f1-score$', fontsize=16)
plt.show()

# print ('best:', best)
#
# print( 'trials:')
# for trial in trials.trials[:2]:
#     print (trial)
