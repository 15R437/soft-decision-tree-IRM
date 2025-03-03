"""Utility functions helpful for IRM training"""

import torch
from torch import nn
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

""" 
note that sklearn.DecisionTreeClassifier orders nodes s.t. you always go down the left child until you reach a leaf
and then you try the 'next available' right child.

we adopt a full-tree ordering that is depth first and prioritises left nodes so that the root node (d=0) has id 0,
its left and right children (d=1) have id 1 and 2 respectively; their children (d=2) have id 3,4 and 5,6 respectively etc:
                                                    0
                                            1               2
                                        3       4       5       6
"""
class MinMaxNormalisation(nn.Module):
    def __init__(self, feature_range=(0, 1)):
        super(MinMaxNormalisation, self).__init__()
        self.min_val = None
        self.max_val = None
        self.is_active = False
        self.feature_range = feature_range
    
    def set_vals(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        if min_val != None and max_val != None:
            self.is_active = True

    def deactivate(self):
        self.is_active = False

    def forward(self, x):
        if not self.is_active:
            return x
        
        degenerate = (self.min_val==self.max_val).float() #1 iff min_val=max_val else 0
        normalised_x = (x - self.min_val) / ((self.max_val - self.min_val)+degenerate) #no division by 0
        normalised_x = normalised_x * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return normalised_x
    
class FeatureMask(nn.Module):
    def __init__(self,input_dim,init_weight=None,random_seed=None):
        super(FeatureMask,self).__init__()
        if random_seed!=None:
            torch.manual_seed(random_seed)
        self.input_dim = input_dim
        if init_weight==None:
            self.weight = nn.Parameter(torch.rand(input_dim))
        else:
            try:
                assert init_weight.shape[0]==input_dim
                self.weight = nn.Parameter(init_weight)
            except:
                raise Exception(f"Expected init_weight to be None or else a 1D tensor of size ({input_dim,}) since input_dim={input_dim}")
        self.layers = [self]

    def forward(self,x):
        try:
            assert x.shape[-1] == self.input_dim
        except:
            raise Exception(f"Expected x to be a tensor of shape (batch_size, {self.input_dim}).")
        return nn.ReLU()(self.weight*x)

class MLPFeaturiser(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dims):
        super(MLPFeaturiser,self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim,hidden_dims[0]))
        self.layers.append(nn.ReLU())
        try:
            assert len(hidden_dims)==len(hidden_dims)
            for i in range(1,len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i],hidden_dims[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dims[-1],output_dim))
            self.layers.append(nn.ReLU()) #latent representation is non-negative
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

class SoftTreeArgs():
    def __init__(self,input_dim,output_dim,
                 batch_size=16,device='mps',lmbda=0.1,max_depth=3,lr=0.001,momentum=0.1,log_interval=1,phi=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.lmbda = lmbda
        self.max_depth = max_depth
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        if phi == None:
            self.phi = FeatureMask(input_dim)
        else:
            self.phi = phi

class PaddedTree():
    def __init__(self,feature,threshold,max_depth,dummy_nodes):
        self.feature = feature
        self.threshold = threshold
        self.max_depth = max_depth
        self.dummy_nodes = dummy_nodes
        self.node_count = len(feature)

class DummyNode():
    def __init__(self,feature,threshold):
        self.feature = feature
        self.threshold = threshold

def add_dummy_nodes(tree_clf:DecisionTreeClassifier):
    """
    This function pads a hard decision tree by adding dummy nodes s.t. 
    if the tree has max_depth d, then it will return a tree with 2**(d+1) -1 nodes.
      
    Leaf nodes in the original tree at depth <d will become dummy nodes which
    inherit the feature and threshold values of their parents and pass these on
    to their children until we reach the max_depth d.
    """
    
    tree = tree_clf.tree_
    max_depth = tree.max_depth
    feature = [] #preferred ordering
    threshold = [] #preferred ordering

    node_id = [0] #tracks the node in DecisionTreeClassifier.tree_ object. should be ordered in such a way that the nodes they point to follow our preferred ordering
    dummy_nodes = [] #tracks whether a node is dummy (1) or not (0)
    while len(feature) < (2**(max_depth+1))-1:
        node = node_id.pop(0)
        if type(node) == DummyNode:
            feature.append(node.feature)
            threshold.append(node.threshold)
            node_id.append(node) #left child
            node_id.append(node) #right child
            dummy_nodes.append(1)
            continue
        ft,th = tree.feature[node],tree.threshold[node]
        if ft!= -2: #inner node
            feature.append(ft)
            threshold.append(th)
            node_id.append(tree.children_left[node]) #left child
            node_id.append(tree.children_right[node]) #right child
            dummy_nodes.append(0)
        else: #leaf node 
            pos = len(feature) #position of current node in our preferred ordering
            if pos==0: #root node is leaf
                return PaddedTree(feature,threshold,max_depth,dummy_nodes)
            par_pos = (pos-1)//2 #position of parent node in our preferred ordering
            dummy = DummyNode(feature[par_pos],threshold[par_pos])

            feature.append(dummy.feature)
            threshold.append(dummy.threshold)
            node_id.append(dummy) # left child
            node_id.append(dummy) # right child
            dummy_nodes.append(1)

    return PaddedTree(feature,threshold,max_depth,dummy_nodes)

def decision_tree_penalty(soft_tree, X, y, depth_discount_factor=1):
    """this computes an optimal hard decision tree given data and target and then computes
    the L2 distance between the non-dummy weightsbetween this optimal tree and soft_tree. """
    tree_classifier = DecisionTreeClassifier(max_depth=soft_tree.args.max_depth,random_state=0) #important to fix random state so in the case where features tie, we always get the same tree structure.
    if type(X) == torch.Tensor:
        X_ = X.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y_  = y.detach().cpu().numpy()
    
    num_features = X.shape[1]
    tree_classifier.fit(X_,y_)
    try:
        padded_tree = add_dummy_nodes(tree_classifier)
    except:
        import pdb; pdb.set_trace()

    W_opt = [] #the optimal coefficients at each node according to tree_classifier
    W = [] #the coefficients of soft_tree collected at those nodes which exist in tree_classifier
    discount = []
    if padded_tree.node_count ==0:
        return torch.tensor(0.).to(soft_tree.args.device)
    
    for node in range(padded_tree.node_count):
        if padded_tree.dummy_nodes[node]:
            continue
        W_opt.append(torch.tensor([1. if i==padded_tree.feature[node] else 0. for i in range(num_features)]
                      + [-padded_tree.threshold[node]]).float())
        W.append(torch.cat([soft_tree.module_list[node].weight,
                  + soft_tree.module_list[node].bias.view(-1,1)],dim=1)[0])
        
        discount.append(torch.tensor(depth_discount_factor**(-math.floor(math.log(node+1,2)))).to(soft_tree.args.device))

    W_opt = torch.stack(W_opt).to(soft_tree.args.device)
    W = torch.stack(W).to(soft_tree.args.device)
    l2_dist = torch.sum((W_opt-W)**2,dim=1) #node-wise distance

    return torch.mean(torch.stack(discount)*l2_dist)

def max_one_regularisation(weights):
    """Encourages one-hot or near-one-hot vectors."""
    loss = torch.tensor(0.).to(weights.device)
    for w in weights: #iterate over all rows
        abs_w = torch.abs(w)
        max_val = torch.max(abs_w)
        loss += torch.sum((abs_w - max_val)**2) # Penalise differences from the maximum value
        loss += torch.abs(torch.sum(abs_w) -1 ) # Penalise if sum of absolute value is not 1
    return loss

def phi_regularisation(weights):
    """Encourages each row to pick different features (in a particular order??)"""
    loss = torch.tensor(0.).to(weights.device)
    for row,w in enumerate(weights):
        if row == 0:
            continue
        else:
            for col,v in enumerate(w):
                loss+= 1

    pass

def feature_selector(weights):
    """
    """
    output = []
    for w in weights:
        output.append([torch.relu(torch.max(w)-0.5) if i==torch.argmax(w) else 0. for i in range(len(w))])
    return torch.tensor(output)