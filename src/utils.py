"""Utility functions helpful for IRM training"""

import torch
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
class SoftTreeArgs():
    def __init__(self,input_dim,output_dim,
                 batch_size=16,device='cpu',lmbda=1,max_depth=3,lr=0.01,momentum=0.1,log_interval=5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.lmbda = lmbda
        self.max_depth = max_depth
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        
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
            depth = math.floor(math.log(pos+1,2)) #current depth in tree
            par_pos = int((2**(depth-1))-1 + math.floor((pos-(2**depth)+1)/2)) #position of parent node in our preferred ordering - can also do par_pos = (pos-1)//2
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
    tree_classifier = DecisionTreeClassifier(max_depth=soft_tree.args.max_depth)
    if type(X) == torch.Tensor:
        X_ = X.detach().numpy()
    if type(y) == torch.Tensor:
        y_  = y.detach().numpy()
    
    num_features = X.shape[1]
    tree_classifier.fit(X_,y_)
    padded_tree = add_dummy_nodes(tree_classifier)

    W_opt = [] #the optimal coefficients at each node according to tree_classifier
    W = [] #the coefficients of soft_tree collected at those nodes which exist in tree_classifier
    discount = []

    for node in range(padded_tree.node_count):
        if padded_tree.dummy_nodes[node]:
            continue
        W_opt.append(torch.tensor([1. if i==padded_tree.feature[node] else 0. for i in range(num_features)]
                      + [-padded_tree.threshold[node]]))
        W.append(torch.cat([soft_tree.module_list[node].weight,
                  + soft_tree.module_list[node].bias.view(-1,1)],dim=1)[0])
        
        discount.append(torch.tensor(depth_discount_factor**(-math.floor(math.log(node+1,2)))))

    W_opt = torch.stack(W_opt)
    W = torch.stack(W)
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