import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from utils import add_dummy_nodes, decision_tree_penalty,SoftTreeArgs
from src.model import SoftDecisionTree

NUM_ENVS = 10
rand_X = [torch.randn(100,4) for i in range(NUM_ENVS)]
rand_y = [torch.Tensor([1 if (x[0]>0.5 and x[1]<0.5 and x[3]<0.5) or 
                    (x[0]<0.5 and x[2]>0.5)else 0 for x in X]) for X in rand_X]
X = torch.stack(rand_X)
y = torch.stack(rand_y).long()
envs = [DataLoader(TensorDataset(X[i,:,:],y[i,:]),batch_size=20) for i in range(NUM_ENVS)]
erm_dataloader = DataLoader(TensorDataset(torch.cat(rand_X,dim=0),torch.cat(rand_y,dim=0).long()),batch_size=200)

args = SoftTreeArgs(input_dim=4,output_dim=2)
soft_tree1 = SoftDecisionTree(args)
soft_tree2 = SoftDecisionTree(args)
#import pdb; pdb.set_trace()

NUM_EPOCHS = 100
print("ERM Training")
for epoch in range(1,NUM_EPOCHS):
    soft_tree1.train_erm(erm_dataloader,epoch)
print("IRM Training")
for epoch in range(1,NUM_EPOCHS+1):
    soft_tree2.train_irm(envs,epoch,penalty_anneal_iters=50,l1_weight=0.01,max_one_weight=0.1)

#test soft_tree.phi to check that it learns properly
