import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from utils.general import SoftTreeArgs
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from generate_data import generate_and_save,func_sigmoid
from utils.general import FeatureMask

#LOADING DATA
curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir,"data/sigmoid_data.pickle")
if os.path.exists(file_path):
    with open(file_path,'rb') as file:
        train_data,test_data,best_params = pickle.load(file)
else:
    train_envs = [0.1,0.2,0.3] #if we bring an umbrella, we are likely to also bring a raincoat
    test_envs = [0.9] # if we bring an umbrella, we are unlikely to also bring a raincoat
    param_grid = {
        'penalty_anneal_iters': [50,95],
        'penalty_weight': [1],
        'l1_weight_feat': [100],
        'l1_weight_tree': [10],
        'lr':[0.1],
        'num_epochs':[100],
        'lmbda': [0.1],
        'depth_discount_factor': [1]}

    train_data,test_data,best_params = generate_and_save(n_train_samples=10000,n_test_samples=1000,train_envs=train_envs,test_envs=test_envs,
          y_func=func_sigmoid,param_grid=param_grid,save_as="data/sigmoid_data.pickle",batch_size=1000,random_seed=0)

train_data_irm = train_data['irm_envs']
train_data_erm = train_data['erm_loader']
X_train_raw,y_train_raw = train_data['raw_data']

test_loader = test_data['erm_loader']
X_test_raw,y_test_raw = test_data['raw_data']

best_lr = best_params['lr'] #0.1
best_l1_feat = best_params['l1_weight_feat'] #100
best_l1_tree = best_params['l1_weight_tree'] #10
best_penalty_weight = best_params['penalty_weight'] #1
best_penalty_anneal = best_params['penalty_anneal_iters'] #95
tree_args = SoftTreeArgs(input_dim=3,output_dim=2,batch_size=1000,lr=best_lr,max_depth=3,log_interval=1)