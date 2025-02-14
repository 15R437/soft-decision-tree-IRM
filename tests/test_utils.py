import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import add_dummy_nodes, decision_tree_penalty
from src.model import SoftDecisionTree
from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np

def id_to_pos(tree_clf):
    """
    This returns a function that takes the sklearn node id 
    and returns the node position according to our preferred
    ordering.
    """
    tree = tree_clf.tree_
    inj = {0:0} #the root node always has index 0 in both orderings

    PrevNodeIsLChildLeaf = False
    PrevNodeIsRChildLeaf = False
    IsCousinNode = False

    for id in range(1,tree.node_count):
        ft = tree.feature[id]
        if ft!=-2: #inner node
            if PrevNodeIsRChildLeaf: #current node is an uncle of the previous node
                inj[id] = inj[id-1]//2
                IsCousinNode = True 

            elif IsCousinNode: #current node is a cousin to the node before the previous node
                inj[id] = inj[id-2]+1
                IsCousinNode = False

            elif PrevNodeIsLChildLeaf: #current node is a sibling to previous node
                inj[id] = inj[id-1]+1

            else:
                inj[id] = 2*inj[id-1]+1 #we can prove inductively that under our preferred ordering, the left child of node k is node 2k+1
            
            PrevNodeIsLChildLeaf = False
            PrevNodeIsRChildLeaf = False
            
        else: #leaf node
            if PrevNodeIsLChildLeaf:
                inj[id] = inj[id-1]+1
                PrevNodeIsRChildLeaf = True
                PrevNodeIsLChildLeaf = False

            elif IsCousinNode:
                inj[id] = inj[id-2]+1
                PrevNodeIsLChildLeaf = True

            elif PrevNodeIsRChildLeaf: #curent leaf is an uncle and so is a right child
                inj[id] = inj[id-1]//2

            else:
               inj[id] = 2*inj[id-1]+1
               PrevNodeIsLChildLeaf = True

    def func(id:int):
        return inj[id]
    
    return func

#10 datasets of size 100
rand_X = [np.random.randn(100,4) for i in range(10)]
rand_y = [np.array([1. if (x[0]>0.5 and x[1]<0.5 and x[3]<0.5) or 
                    (x[0]<0.5 and x[2]>0.5)else 0. for x in X]) for X in rand_X]
rand_trees = [DecisionTreeClassifier(max_depth=3).fit(X,y) for X,y in zip(rand_X,rand_y)]

@pytest.mark.parametrize("tree_clf",rand_trees)
def test_add_dummy_nodes(tree_clf):
    tree = tree_clf.tree_
    padded_tree = add_dummy_nodes(tree_clf)

    true_node_count = 2**(tree.max_depth+1)-1
    num_leaves = sum([1 if tree.feature[i]==2 else 0 for i in range(tree.node_count)])
    
    assert padded_tree.max_depth == tree.max_depth
    assert padded_tree.node_count == true_node_count
    assert len(padded_tree.threshold) == true_node_count
    assert sum(padded_tree.dummy_nodes) == num_leaves + (true_node_count - tree.node_count)
    for node in range(true_node_count):
        assert padded_tree.feature == 2
        assert padded_tree.threshold == 3
