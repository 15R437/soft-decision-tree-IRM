import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from src.utils import SoftTreeArgs
import numpy as np
import tuning
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch import nn, optim, autograd

#MODEL CLASSES

class MLP(nn.Module):
    def __init__(self,hidden_dim=390,grey_scale=False):
        super(MLP,self).__init__()
        self.grey_scale = grey_scale
        if grey_scale:
            self.lin1 = nn.Linear(14*14,hidden_dim)
        else:
            self.lin1 = nn.Linear(2*14*14,hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,hidden_dim)
        self.lin3 = nn.Linear(hidden_dim,1)

        for lin in [self.lin1,self.lin2,self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
    
    def forward(self,x):
        if self.grey_scale:
            x = x.view(-1,14*14)
        else:
            x = x.view(-1,2*14*14)
        x = nn.ReLU()(self.lin1(x))
        x = nn.ReLU()(self.lin2(x))
        x  = self.lin3(x)
        return x

class MinMaxNormalisation(nn.Module):
    def __init__(self, feature_range=(0, 1)):
        super(MinMaxNormalisation, self).__init__()
        self.min_val = None
        self.max_val = None
        self.feature_range = feature_range
    
    def set_vals(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        if self.min_val == self.max_val:
            return torch.full_like(x, self.feature_range[0]) #return the min range if min and max are equal.

        normalised_x = (x - self.min_val) / (self.max_val - self.min_val)
        normalised_x = normalised_x * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return normalised_x

    
class MLPFeaturiser(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dims):
        super(MLPFeaturiser,self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,hidden_dims[0]))
        self.layers.append(nn.ReLU())
        try:
            assert len(hidden_dims)==len(hidden_dims)
            for i in range(1,len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dims[-1],output_dim))
        except:
            try: 
                assert type(hidden_dims)==int
                self.layers.append(nn.Linear(hidden_dims,output_dim))
            except:
                raise Exception(f"Expected hidden_dims to be a list of integer dimensions or else a single integer. Instead got type {type(hidden_dims)}")
            
        self.minmax_norm =MinMaxNormalisation()

    def forward(self,x,minmax_norm=True):
        x = x.view(-1,self.input_dim)
        for layer in self.layers:
            x = layer(x)
        if minmax_norm:
            x = self.minmax_norm(x)
        
        return x

#Data processing
#torch.manual_seed(0)

mnist = datasets.MNIST(root='~/datasets/mnist', train=True,download=True)
train_imgs,train_labels = mnist.data[:50000].float(), mnist.targets[:50000]
val_imgs, val_labels = mnist.data[50000:].float(),mnist.targets[50000:]

#binarising train and val labels
train_labels = (train_labels>4).float()
val_labels = (val_labels>4).float()
def add_noise(labels,e):
    flip = (torch.rand(len(labels))<e).float()
    noisy_labels = torch.abs(flip-labels)
    return noisy_labels

train_labels = add_noise(train_labels,0.25)
val_labels = add_noise(val_labels,0.25)

def make_envs(img_list:list,label_list:list,e_list:list,batch_size=None,random_seed=None):
    """
    This function takes a list of image batches and corresponding labels and environment parameter e and
    returns a list of torch dataloaders each one corresponding to the environment defined by e_list

    """
    try:
        assert len(img_list)==len(label_list)
        assert len(label_list)==len(e_list)
    except:
        raise Exception(f"Expected imgs,labels and e_list to be lists of equal length.")
    
    if random_seed==None:
        pass
    else:
        torch.manual_seed(random_seed)
    
    envs = []
    X = []
    y = []

    for id,e in enumerate(e_list):
        try:
            assert e>= 0 and e<=1
        except:
            raise Exception(f"e should be a list of probabilities. Instead got {e} in position {id}.")
        
        imgs = img_list[id]
        labels = label_list[id]
        if batch_size==None:
            batch_size = len(imgs)

        flip = (torch.rand(len(labels))<e).float()
        colour = torch.abs(flip-labels)
        imgs = imgs.reshape((-1,28,28))[:,::2,::2]
        coloured_imgs = torch.stack([imgs,imgs],dim=1)
        coloured_imgs[torch.tensor(range(len(imgs))),(1-colour).long(),:,:] *= 0
        coloured_imgs /= 255.0
        X.append(coloured_imgs)
        y.append(labels)
        
        envs.append(DataLoader(TensorDataset(coloured_imgs,labels),batch_size=batch_size))
    
    X,y = torch.cat(X),torch.cat(y)
    erm_loader = DataLoader(TensorDataset(X,y))
    return {'irm_envs':envs, 'erm_loader':erm_loader, 'raw_data':(X,y)}

img_list = [train_imgs[::2],train_imgs[1::2]]
label_list = [train_labels[::2],train_labels[1::2]]
train_envs = [0.2,0.1]
val_envs = [0.9]

train_data = make_envs(img_list,label_list,train_envs)
val_loader = make_envs([val_imgs],[val_labels],val_envs)['erm_loader']

irm_envs = train_data['irm_envs']
erm_loader = train_data['erm_loader']
X_raw,y_raw = train_data['raw_data']


tree_args = SoftTreeArgs(input_dim=10,output_dim=2,phi=MLPFeaturiser(input_dim=2*14*14,output_dim=10))
soft_tree = SoftDecisionTree(tree_args)

import pdb; pdb.set_trace()
for epoch in range(1,101):
    #comput min and max value over data set
    min_val = torch.min(soft_tree.phi.forward(X_raw,False),dim=0)[0]
    max_val = torch.max(soft_tree.phi.forward(X_raw,False),dim=0)[0]
    soft_tree.phi.minmax_norm.set_vals(min_val,max_val)
    soft_tree.train_irm(irm_envs,epoch)


"""erm_mlp = MLP(hidden_dim=256).to(device)
irm_mlp = MLP().to(device)
oracle_mlp = MLP(hidden_dim=83,grey_scale=True).to(device)

def mean_nll(logits,y):
    return nn.functional.binary_cross_entropy_with_logits(logits,y)

def mean_acc(logits,y):
    pred = (logits>0).float()
    return (torch.abs((pred-y))).float().mean()

def inv_penalty(logits,y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits*scale,y)
    grad = autograd.grad(loss,[scale],create_graph=True)[0]
    return torch.sum(grad**2)"""