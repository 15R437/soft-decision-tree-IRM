"""Utility functions helpful for IRM training"""
from model import SoftDecisionTree
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
    max_depth = tree.max_dept
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
            par_pos = (2**(depth-1))-1 + math.floor((pos-(2**depth)+1)/2) #position of parent node in our preferred ordering
            dummy = DummyNode(feature[par_pos],threshold[par_pos])

            feature.append(dummy.feature)
            threshold.append(dummy.threshold)
            node_id.append(dummy) # left child
            node_id.append(dummy) # right child
            dummy_nodes.append(1)

    return PaddedTree(feature,threshold,max_depth,dummy_nodes)

def decision_tree_penalty(soft_tree:SoftDecisionTree, X, y, depth_discount_factor=1,pad_tree=True):
    """this computes an optimal hard decision tree given data and target and then computes
    the L2 distance between the non-dummy weightsbetween this optimal tree and soft_tree. """
    tree_classifier = DecisionTreeClassifier(max_depth=soft_tree.args.max_depth)
    if type(X) == torch.Tensor:
        X = X.numpy()
    if type(y) == torch.Tensor:
        y  = y.numpy()
    
    num_features = X.shape[1]
    tree_classifier.fit(X,y)
    padded_tree = add_dummy_nodes(tree_classifier)

    W_opt = [] #the optimal coefficients at each node according to tree_classifier
    W = [] #the coefficients of soft_tree collected at those nodes which exist in tree_classifier
    discount = []

    for node in range(padded_tree.node_count):
        if padded_tree.dummy_nodes[node]:
            continue
        W_opt.append([1. if i==padded_tree.feature[node] else 0. for i in range(num_features)]
                      + [-padded_tree.threshold[node]])
        W.append(torch.cat([soft_tree.module_list[node].weight,
                  + soft_tree.module_list[node].bias.view(-1,1)],dim=1)[0])
        
        discount.append(depth_discount_factor**(-math.floor(math.log(node+1,2))))

    W_opt = torch.Tensor(W_opt)
    W = torch.Tensor(W)
    l2_dist = torch.sum((W_opt-W)**2,dim=1) #node-wise distance

    return torch.mean(discount*l2_dist)
