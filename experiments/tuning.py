import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from src.utils import SoftTreeArgs
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import itertools 
from sklearn.model_selection import KFold,train_test_split  # For cross-validation
from sklearn.metrics import mean_squared_error # Example metric, use your own
from sklearn.preprocessing import MinMaxScaler

class DataObject():
    def __init__(self,data,test_data=None,scaler=MinMaxScaler()):
        if not test_data is None:
            self.test_data = DataObject(test_data,scaler=scaler)
        else:
            self.test_data = None

        if type(data)==DataLoader:
            self.training_type = 'erm'
            X,y = next(iter(data))
            while True:
                try:
                    X_,y_ = next(iter(data))
                    X = torch.cat([X,X_],dim=0)
                    y =  torch.cat([y,y_],dim=0)
                except StopIteration:
                    break
            self.X = np.array(X)
            self.y = np.array(y)

        elif type(data)==list:
            self.training_type = 'irm'
            X_envs = []
            y_envs = []
            for id,loader in enumerate(data):
                try:
                    X,y = next(iter(loader))
                except:
                    raise Exception(f"Expected list of data loaders. Instead got {type(loader)} at index {id} ")
                while True:
                    try:
                        X_,y_ = next(iter(data))
                        X = torch.cat([X,X_],dim=0)
                        y = torch.cat([y,y_],dim=0)
                    except StopIteration:
                        break
                X_envs.append(X)
                y_envs.append(y)

            self.X = np.array(X_envs) #3d array whose rows are environments. Assumes that...
            self.y = np.array(y_envs) #..each environment has the same number of data points.

        else:
            raise Exception(f"""Expected either a Pytorch DataLoader object or a list of Pytorch DataLoader objects.
                             Instead got {type(data)}""")
        if not scaler is None:
            self.X_scaled = scaler.fit_transform(self.X)

def tune(input_dim,output_dim,data_object:DataObject,param_grid:dict,scaler=MinMaxScaler(),k=5):
    if scaler==None:
        X_train_scaled = data_object.X
    else:
        X_train_scaled = scaler.fit_transform(data_object.X)
    
    y_train = data_object.y
    input_dim = X_train_scaled.shape[-1]
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    best_acc = 0
    best_params = {}

    #kfold validation
    kf = KFold(n_splits=k,shuffle=True,random_state=1)
    print("""penalty_anneal_iters \t penalty_weight \t l1_weight_feat \t l1_weight_tree \t val_accuracy \t NewBest""")
    for combination in combinations:
        params = dict(zip(keys,combination))
        cv_scores = []
        NewBest = False
        for train_id,val_id in kf.split(X_train_scaled):
            X_train_fold, X_val_fold = X_train_scaled[train_id],X_train_scaled[val_id]
            y_train_fold, y_val_fold = y_train[train_id],y_train[val_id]
            if data_object.training_type=='irm': #each row of X_train_fold is an environment
                penalty_anneal_iters = params['penalty_anneal_iters']
                penalty_weight = params['penalty_weight']
                l1_weight_feat = params['l1_weight_feat']
                l1_weight_tree = params['l1_weight_tree']
                depth_discount = params['depth_discount_factor']
                try:
                    lmbda = params['lmbda']
                except:
                    lmbda = 0.1
                try:
                    lr = params['lr']
                except:
                    lr = 0.01
                try:
                    num_epochs = params['num_epochs']
                except:
                    num_epochs = 100
                try:
                    max_one_weight = params['max_one_weight']
                except:
                    max_one_weight = 0

                envs = [DataLoader(TensorDataset(torch.tensor(X),torch.tensor(y)))for X,y in zip(X_train_fold,y_train_fold)]
                tree_args = SoftTreeArgs(input_dim=input_dim,output_dim=output_dim,
                                         lr=lr,lmbda=lmbda)
                soft_tree = SoftDecisionTree(tree_args)
                for epoch in range(1,101):
                    soft_tree.train_irm(envs,epoch,print_progress=False,return_stats=False,penalty_weight=penalty_weight,depth_discount_factor=depth_discount,
                                        penalty_anneal_iters=penalty_anneal_iters, l1_weight_feat=l1_weight_feat,l1_weight_tree=l1_weight_tree)
                
                num_envs,num_data = y_val_fold.shape
                target_one_hot = torch.zeros(num_envs,num_data,tree_args.output_dim)
                target_one_hot.scatter_(2,y_val_fold.view(num_envs,num_data,1),1.)
                data = torch.cat(list(torch.tensor(X_val_fold)))
                target = torch.cat(list(torch.tensor(y_val_fold)))
                _,output = soft_tree.cal_loss(data,target_one_hot)
                pred = output.data.max(1)[1] # get the index of the max log-probability
                correct = pred.eq(target).cpu().sum()
                accuracy = 100. * correct / len(data)
                cv_scores.append(accuracy)
            
            elif data_object.training_type=='erm':
                pass
        
        mean_cv_score = np.mean(cv_scores)
        if mean_cv_score > best_acc:  # Assuming minimizing a loss
            best_acc = mean_cv_score
            best_params = params
            NewBest = True
        
        print(f"""{penalty_anneal_iters} \t {penalty_weight} \t {l1_weight_feat} \t {l1_weight_tree} \t {accuracy} \t {NewBest} """)
    pass

# 1. Define the Hyperparameter Grid:
param_grid = {
    'l1_strength': [0.001, 0.01, 0.1, 1.0],  # L1 regularization strength
    'other_reg_strength': [0.001, 0.01, 0.1, 1.0],  # Strength of your other regularization term
    'learning_rate': [0.001, 0.01, 0.1]  # Example: learning rate if you're using gradient descent
}