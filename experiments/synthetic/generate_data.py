import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.dirname(script_dir)
working_dir = os.path.dirname(experiments_dir)
utils_dir = os.path.join(working_dir,'utils')
sys.path.append(utils_dir)

import utils.tuning as tuning

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

def func_sigmoid(w,b):
    def sigmoid(x,w=w):
        x = x.view(1,-1)
        w = w.view(x.shape[1],1)
        return torch_bernoulli(torch.sigmoid(x@w+b).view(-1))
    return sigmoid

def make_environments(n_samples:int,e_list:list,y_func,scaler=MinMaxScaler(),batch_size=None,random_seed=None,spurious_sigmoid_param=None):
    """we initialise a spurious variable x[2] = y and then flip its value w.p. e where 
    e defines the environment. x[0] is sampled uniformly from [0,1) and x[1] ~ Bernoulli(1-x[0]).

    These can be interpreted e.g. as x[0] = probability of rain, x[1] = whether or not one cycles to work,
    y = whether or not one brings an umbrella to work and x[2] = whether or not one brings a rain coat.
    """
    if n_samples == None:
        return None
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
        x_1 = torch_bernoulli(0.5,size=x_0.size()) #changed the Bernoulli parameter from 1-x_0 to a constant p=.5
        y = torch.cat([y_func(x) for x in torch.cat([x_0,x_1],dim=1)],dim=0)
        if spurious_sigmoid_param==None:
            x_2 = (y-torch_bernoulli(e,size=y.size()[0])).abs().view(-1,1)
        else:
            w,b = spurious_sigmoid_param
            w,b = (2*e-1)*w,(2*e-1)*b
            x_2 = torch.cat([func_sigmoid(w,b)(y_.float()) for y_ in y]).view(-1,1)

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

def generate_and_save(n_train_samples,n_test_samples,train_envs,test_envs,y_func,param_grid={},save_as=None,tune=True,batch_size=None,random_seed=None,spurious_sigmoid_param=None):
    train_data = make_environments(n_samples=n_train_samples,e_list=train_envs,y_func=y_func,batch_size=batch_size,random_seed=random_seed,spurious_sigmoid_param=spurious_sigmoid_param)
    test_data = make_environments(n_samples=n_test_samples,e_list=test_envs,y_func=y_func,batch_size=batch_size,random_seed=None,spurious_sigmoid_param=spurious_sigmoid_param) #test data is random

    if train_data!=None:
        train_data_irm = train_data['irm_envs']

    #hyperparemeter tuning
    if tune:
        data_object = tuning.DataObject(train_data_irm)
        best_params = tuning.tune(3,2,data_object,param_grid,k=3)
    else:
        best_params = None

    train_test_tune_data = (train_data,test_data,best_params)
    if save_as==None:
        return train_test_tune_data

    #saving train,test and tuning data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, save_as)
    

    with open(file_path,"wb") as file:
        pickle.dump(train_test_tune_data,file)
    
    return train_test_tune_data