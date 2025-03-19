from utils.featuriser import FeatureMask
from sklearn.tree import DecisionTreeClassifier
import torch
import math

class SoftTreeArgs():
    def __init__(self,input_dim,output_dim,
                 batch_size=16,device='mps',lmbda=0.1,max_depth=3,lr=0.001,momentum=0.1,log_interval=1,phi=None,tree_weights=None,beta=None,leaf_dist=None,dtype=torch.float16):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.lmbda = lmbda
        self.max_depth = max_depth
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        self.tree_weights = tree_weights
        self.beta = beta
        self.leaf_dist = leaf_dist
        self.dtype=dtype
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
        X = X.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y  = y.detach().cpu().numpy()
    
    num_features = X.shape[1]
    tree_classifier.fit(X,y)
    padded_tree = add_dummy_nodes(tree_classifier)

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
        try:
            W.append(torch.cat([soft_tree.module_list[node].weight.view(1,-1),
                  + soft_tree.module_list[node].bias.view(1,1)],dim=1)[0])
        except:
            import pdb; pdb.set_trace()
        
        discount.append(torch.tensor(depth_discount_factor**(-math.floor(math.log(node+1,2)))).to(soft_tree.args.device))

    W_opt = torch.stack(W_opt).to(soft_tree.args.device)
    W = torch.stack(W).to(soft_tree.args.device)
    l2_dist = torch.sum((W_opt-W)**2,dim=1) #node-wise distance

    return torch.mean(torch.stack(discount)*l2_dist)

def new_tree_penalty(soft_tree,X,y,num):
    """this computes an optimal hard decision tree given data and target and then computes
    the soft_tree loss (over X) wrt target values computed from the hard tree. """
    tree_classifier = DecisionTreeClassifier(max_depth=soft_tree.args.max_depth,random_state=0) #important to fix random state so in the case where features tie, we always get the same tree structure.
    if type(X) == torch.Tensor:
        X = X.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y  = y.detach().cpu().numpy()
    
    tree_classifier.fit(X,y)
    y_pred = tree_classifier.predict_proba(X)
    if num==5:
        #import pdb; pdb.set_trace()
        pass
    X = torch.tensor(X).to(soft_tree.args.device)
    y = torch.tensor(y_pred).to(soft_tree.args.device)
    batch_size = y.shape[0]
    if not batch_size == soft_tree.args.batch_size:
        soft_tree.define_extras(batch_size)
    loss,output,C = soft_tree.cal_loss(X,y,include_featuriser=False)
    #TQ = torch.bmm(y.clone().view(batch_size, 1, soft_tree.args.output_dim).to(output.dtype), torch.log(output).clone().view(batch_size, soft_tree.args.output_dim, 1)).view(-1,1)
    #max_nll_loss = torch.mean(-1.*TQ)
    return loss