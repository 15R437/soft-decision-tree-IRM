import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from src.utils import SoftTreeArgs
from src.utils import MLPFeaturiser
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
        labels = label_list[id].long()
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
    
    X,y = torch.cat(X),torch.cat(y).long()
    erm_loader = DataLoader(TensorDataset(X,y))
    return {'irm_envs':envs, 'erm_loader':erm_loader, 'raw_data':(X,y)}

img_list = [train_imgs[::2],train_imgs[1::2]]
label_list = [train_labels[::2],train_labels[1::2]]
train_envs = [0.2,0.1]
val_envs = [0.9]

train_data = make_envs(img_list,label_list,train_envs,random_seed=0,batch_size=1000)
val_loader = make_envs([val_imgs],[val_labels],val_envs)['erm_loader']


irm_envs = train_data['irm_envs']
erm_loader = train_data['erm_loader']
X_raw,y_raw = train_data['raw_data']

#tuning
data_object = tuning.DataObject(irm_envs)
param_grid = {
    'penalty_anneal_iters': [50,95],
    'penalty_weight': [0.1,1,10],
    'l1_weight_feat': [0.001,0.01,0.1],
    'l1_weight_tree': [0.01,0.1,1],
    'lr':[0.05,0.1],
    'num_epochs':[100],
    'lmbda': [0.1],
    'depth_discount_factor': [1]

}
featuriser = MLPFeaturiser(input_dim=2*14*14,output_dim=10,hidden_dims=[390])
best_params = tuning.tune(10,2,data_object,param_grid,k=2,scaler=None,phi=featuriser)
import pdb; pdb.set_trace()

best_lr = 0.05 #0.1
best_l1_feat = 0.001 #100
best_l1_tree = 1 #10
best_penalty_weight = 0.1 #1
best_penalty_anneal = 95 #95

tree_args = SoftTreeArgs(input_dim=10,output_dim=2,phi=featuriser,lr=best_lr)
soft_tree = SoftDecisionTree(tree_args)

#X_raw = X_raw.to(tree_args.device)
#y_raw = y_raw.to(tree_args.device)

for epoch in range(1,101):
    soft_tree.train_irm(irm_envs,epoch,penalty_anneal_iters=best_penalty_anneal,penalty_weight=best_penalty_weight,
                        l1_weight_feat=best_l1_feat,l1_weight_tree=best_l1_tree)
    #import pdb; pdb.set_trace()


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