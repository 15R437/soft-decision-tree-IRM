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
        x_axis,sigmoid_weights,train_data_list,test_data_list,best_params_list = pickle.load(file)
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
    
    x_axis = [k for k in range(10)]
    sigmoid_weights = [(k*torch.tensor([2.,-2.]),k*torch.tensor(-1.)) for k in x_axis]
    train_data_list,test_data_list,best_params_list = [],[],[]

    for w,b in sigmoid_weights:
        train_data,test_data,best_params = generate_and_save(n_train_samples=10000,n_test_samples=1000,train_envs=train_envs,test_envs=test_envs,
          y_func=func_sigmoid(w,b),param_grid=param_grid,save_as=None,batch_size=1000,random_seed=0)
        
        train_data_list.append(train_data)
        test_data_list.append(test_data)
        best_params_list.append(best_params)
    
    with open(file_path,"wb") as file:
        pickle.dump((x_axis,sigmoid_weights,train_data_list,test_data_list,best_params_list),file)

def experiment(num_trials):
    print("Running Experiment..")
    irm_acc = []
    erm_acc = []
    hard_tree_acc = []
    random_forest_acc = []

    for train_data,test_data,best_params in zip(train_data_list,test_data_list,best_params_list):

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

            
        hard_tree,random_forest = DecisionTreeClassifier(max_depth=tree_args.max_depth,random_state=0), RandomForestClassifier(n_estimators=10,random_state=0)
        hard_tree.fit(X_train_raw,y_train_raw)
        random_forest.fit(X_train_raw,y_train_raw)

        tree_preds,forest_preds = hard_tree.predict(X_test_raw),random_forest.predict(X_test_raw)
        tree_acc, forest_acc = accuracy_score(y_test_raw,tree_preds),accuracy_score(y_test_raw,forest_preds)
        hard_tree_acc.append(tree_acc)
        random_forest_acc.append(forest_acc)

        erm_test_accuracy = []
        irm_test_accuracy = []

        for trial in range(num_trials):
            soft_tree_erm = SoftDecisionTree(tree_args)
            soft_tree_irm = SoftDecisionTree(tree_args)
            print(f"ERM Training {trial+1}")
            for epoch in range(1,51):
                soft_tree_erm.train_erm(train_data_erm,epoch,l1_weight_tree=best_l1_tree)
            print(f"IRM Training {trial+1}")
            for epoch in range(1,101):
                soft_tree_irm.train_irm(train_data_irm,epoch,penalty_anneal_iters=best_penalty_anneal,l1_weight_feat=best_l1_feat,
                                        l1_weight_tree=best_l1_tree,penalty_weight=best_penalty_weight)
            
            erm_test_accuracy.append(soft_tree_erm.test_(test_loader,print_result=False,return_acc=True))
            irm_test_accuracy.append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))
        
        irm_acc.append(np.mean(irm_test_accuracy))
        erm_acc.append(np.mean(erm_test_accuracy))
    
    plt.plot(x_axis,irm_acc,label="soft tree (irm)")
    plt.plot(x_axis,erm_acc,label="soft tree (erm)")
    plt.plot(x_axis,hard_tree_acc,label="hard tree")
    plt.plot(x_axis,random_forest_acc,label="random forest")