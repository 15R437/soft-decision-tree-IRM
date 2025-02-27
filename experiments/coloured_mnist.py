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


device = 'mps'
torch.manual_seed(0)

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

def make_env(imgs,labels,flip_prob,debug=False):
    torch.cuda.empty_cache()
    print(f'Creating Environment e={flip_prob}...')
    flip = (torch.rand(len(labels))<flip_prob).float()
    colour = torch.abs(flip-labels)
    imgs = imgs.reshape((-1,28,28))[:,::2,::2]
    if debug:
        import pdb; pdb.set_trace()
    coloured_imgs = torch.stack([imgs,imgs],dim=1)
    coloured_imgs[torch.tensor(range(len(imgs))),(1-colour).long(),:,:] *= 0
    coloured_imgs /= 255.0
    labels = labels.reshape(-1,1)
    return ((coloured_imgs).to(device),labels.to(device))

envs = [
    make_env(train_imgs[::2],train_labels[::2],0.2),
    make_env(train_imgs[1::2],train_labels[1::2],0.1),
    make_env(val_imgs,val_labels,0.9)
]

print(f'Environments Created!')
#Builiding a model

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
import torch
import torch.nn as nn

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

    
class Featuriser(nn.Module):
    def __init__(self,hidden_dim=390,feat_dim=10,grey_scale=False):
        super(MLP,self).__init__()
        self.grey_scale = grey_scale
        if grey_scale:
            self.lin1 = nn.Linear(14*14,hidden_dim)
        else:
            self.lin1 = nn.Linear(2*14*14,hidden_dim)
        self.lin2 = nn.Linear(hidden_dim,feat_dim)
        self.minmax_norm =MinMaxNormalisation()

    def forward(self,x,minmax_norm=True):
        if self.grey_scale:
            x = x.view(-1,14*14)
        else:
            x = x.view(-1,2*14*14)
        x = nn.ReLU()(self.lin1(x))
        x = nn.ReLU()(self.lin2(x))
        if minmax_norm:
            x = self.minmax_norm(x)
        
        return x

phi = Featuriser() 
tree_args = SoftTreeArgs(input_dim=10,output_dim=2,phi=phi)
soft_tree = SoftDecisionTree(tree_args)

for epoch in range(1,101):
    #comput min and max value over data set
    min_val = torch.min(soft_tree.phi.forward(x,False))
    max_val = torch.max(soft_tree.phi.forward(x,False))
    soft_tree.phi.minmax_norm.set_vals(min_val,max_val)

    #train with min max values
    soft_tree.train_irm(envs,epoch)

erm_mlp = MLP(hidden_dim=256).to(device)
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
    return torch.sum(grad**2)