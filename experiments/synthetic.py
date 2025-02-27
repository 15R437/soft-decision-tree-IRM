import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from src.utils import SoftTreeArgs
import numpy as np
import tuning
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Deterministic Function
def func(x):
    if x[0]>0.5 and x[1]<0.5 and x[3]<0.5:
        return 1
    elif x[0]<0.5 and x[2]>0.5:
        return 1
    else:
        return 0
def torch_bernoulli(p,size=None):
    if size==None:
        try:
            size = p.size()
        except:
            raise Exception(f"Expected size argument to be specified or else, p be a tensor from which size can be inferred.")
    return (torch.rand(size)<p).long()

def func_stochastic(x):
    if x[0]>0.5 and x[1]==0: #e.g. rain is likely and I am not cycling to work
        y = torch_bernoulli(0.9,1)
    elif x[0]>0.2 and x[1]==0: #e.g. rain is unlikely and
        y = torch_bernoulli(0.6,1)
    else:
        y = torch_bernoulli(0.1,1)
    return y.long()

def make_environments(n_samples:int,e_list:list,scaler=MinMaxScaler(),batch_size=None,random_seed=None):
    """we initialise a spurious variable x[2] = y and then flip its value w.p. e where 
    e defines the environment. x[0] is sampled uniformly from [0,1) and x[1] ~ Bernoulli(1-x[0]).

    These can be interpreted e.g. as x[0] = probability of rain, x[1] = whether or not one cycles to work,
    y = whether or not one brings an umbrella to work and x[2] = whether or not one brings a rain coat.
    """
    if random_seed==None:
        pass
    else:
        torch.manual_seed(random_seed)
    envs = []
    X_ = []
    y_ = []
    if batch_size==None:
        batch_size = n_samples

    for id,e in enumerate(e_list):
        try:
            assert e>= 0 and e<=1
        except:
            raise Exception(f"e should be a list of probabilities. Instead got {e} in position {id}.")
        x_0 =  torch.rand(n_samples,1)
        x_1 = torch_bernoulli(1-x_0)
        y = torch.cat([func_stochastic(x) for x in torch.cat([x_0,x_1],dim=1)],dim=0)
        x_2 = (y-torch_bernoulli(e,size=y.size()[0])).abs().view(-1,1)
        if scaler==None:
            X = torch.cat([x_0,x_1,x_2],dim=1)
        else:
            X = torch.tensor(scaler.fit_transform(torch.cat([x_0,x_1,x_2],dim=1))).float()

        X_.append(X)
        y_.append(y)

        envs.append(DataLoader(TensorDataset(X,y),batch_size=batch_size))
    
    X_,y_ = torch.cat(X_),torch.cat(y_)
    erm_loader = DataLoader(TensorDataset(X_,y_),batch_size=batch_size)

    return {'irm_envs':envs, 'erm_loader':erm_loader, 'raw_data':(np.array(X_),np.array(y_))}

n_samples = 1000
training_envs = [0.1,0.2,0.3] #if we bring an umbrella, we are likely to also bring a raincoat
test_envs = [0.9] # if we bring an umbrella, we are unlikely to also bring a raincoat

train_data = make_environments(n_samples,training_envs,random_seed=0)
test_data = make_environments(n_samples,test_envs)

train_data_irm = train_data['irm_envs']
train_data_erm = train_data['erm_loader']
X_train_raw,y_train_raw = train_data['raw_data']

test_loader = test_data['erm_loader']
X_test_raw,y_test_raw = test_data['raw_data']

#hyperparemeter tuning
data_object = tuning.DataObject(train_data_irm)
param_grid = {
    'penalty_anneal_iters': [95],
    'penalty_weight': [1,10,50],
    'l1_weight_feat': [10,100],
    'l1_weight_tree': [10,100],
    'lr':[0.01,0.1],
    'num_epochs':[100],
    'lmbda': [0.1],
    'depth_discount_factor': [1]

}
#best_params = tuning.tune(3,2,data_object,param_grid,k=3)
#import pdb; pdb.set_trace()

#soft_tree_irm vs soft_tree_erm vs hard_tree
best_lr = 0.1
best_l1_feat = 100
best_l1_tree = 100
best_penalty_weight = 1
best_penalty_anneal = 95
tree_args = SoftTreeArgs(input_dim=3,output_dim=2,batch_size=100,lr=best_lr,max_depth=3,log_interval=1)

hard_tree,random_forest = DecisionTreeClassifier(max_depth=tree_args.max_depth,random_state=0), RandomForestClassifier(n_estimators=10,random_state=0)
hard_tree.fit(X_train_raw,y_train_raw)
random_forest.fit(X_train_raw,y_train_raw)

tree_preds,forest_preds = hard_tree.predict(X_test_raw),random_forest.predict(X_test_raw)
tree_acc, forest_acc = accuracy_score(y_test_raw,tree_preds),accuracy_score(y_test_raw,forest_preds)

erm_test_accuracy = []
irm_test_accuracy = []

NUM_TRIALS = 10
for trial in range(NUM_TRIALS):
    soft_tree_erm = SoftDecisionTree(tree_args)
    soft_tree_irm = SoftDecisionTree(tree_args)
    print(f"ERM Training {trial+1}")
    for epoch in range(1,51):
        soft_tree_erm.train_erm(train_data_erm,epoch,l1_weight_tree=best_l1_tree)
    print(f"IRM Training {trial+1}")
    for epoch in range(1,101):
        torch.autograd.set_detect_anomaly(True)
        soft_tree_irm.train_irm(train_data_irm,epoch,penalty_anneal_iters=best_penalty_anneal,l1_weight_feat=best_l1_feat,
                                l1_weight_tree=best_l1_tree,penalty_weight=best_penalty_weight)
    
    erm_test_accuracy.append(soft_tree_erm.test_(test_loader,print_result=False,return_acc=True))
    irm_test_accuracy.append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))



print(f"hard tree acc: {100*tree_acc:0.2f}%")
print(f"random forest acc: {100*forest_acc:.2f}%")
print(f"soft tree (erm) acc: {torch.mean(torch.tensor(erm_test_accuracy)).data:0.2f}%")
print(f"soft tree (irm) acc: {torch.mean(torch.tensor(irm_test_accuracy)).data:0.2f}%")