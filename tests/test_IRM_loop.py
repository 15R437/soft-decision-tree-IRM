import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from src.utils import add_dummy_nodes, decision_tree_penalty
from src.model import SoftDecisionTree


class SoftTreeArgs():
    def __init__(self,input_dim,output_dim,
                 batch_size=16,device='cpu',lmbda=1,max_depth=3,lr=0.001,momentum=0.1,log_interval=5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.lmbda = lmbda
        self.max_depth = max_depth
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval

NUM_ENVS = 10
rand_X = torch.stack([torch.randn(100,4) for i in range(NUM_ENVS)])
rand_y = torch.stack([torch.Tensor([1 if (x[0]>0.5 and x[1]<0.5 and x[3]<0.5) or 
                    (x[0]<0.5 and x[2]>0.5)else 0 for x in X]) for X in rand_X]).long()
envs = [DataLoader(TensorDataset(rand_X[i,:,:],rand_y[i,:]),batch_size=20) for i in range(NUM_ENVS)]
#rand_trees = [DecisionTreeClassifier(max_depth=3).fit(X,y) for X,y in zip(rand_X,rand_y)]
args = SoftTreeArgs(input_dim=4,output_dim=2)
soft_tree = SoftDecisionTree(args)
#import pdb; pdb.set_trace()

NUM_EPOCHS = 100
for epoch in range(1,NUM_EPOCHS+1):
    soft_tree.train_irm(envs,epoch,penalty_anneal_iters=20)