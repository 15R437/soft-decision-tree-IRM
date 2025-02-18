import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import add_dummy_nodes, decision_tree_penalty
from src.model import SoftDecisionTree
from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np

def id_to_pos(tree_clf,invert=False):
    """
    This returns a function that takes the sklearn node id 
    and returns the node position according to our preferred
    ordering.
    """
    tree = tree_clf.tree_
    inj = {0:0} #the root node always has index 0 in both orderings

    PrevNodeIsLChildLeaf = False
    PrevNodeIsRChildLeaf = False

    for id in range(1,tree.node_count):
        ft = tree.feature[id]
        if ft!=-2: #inner node
            if PrevNodeIsLChildLeaf: #current node is a sibling to previous node
                inj[id] = inj[id-1]+1
    
            elif PrevNodeIsRChildLeaf: #we find the closest ancestor of the previous node that is a left child and the current node will be its right sibling
                num_gens = 0
                par_pos = (inj[id-1]-2)/2
                IsRChild = 1 if par_pos%1 == 0 else 0
                while IsRChild:
                    num_gens+=1
                    par_pos = (par_pos-2)/2
                    IsRChild = 1 if par_pos%1 == 0 else 0

                curr_pos = inj[id-1]
                for i in range(num_gens-1):
                    curr_pos = (curr_pos-2)/2

                inj[id] = int(curr_pos/2)

            else:
                inj[id] = 2*inj[id-1]+1 #current node is a left child of the previous node
            
            PrevNodeIsLChildLeaf = False
            PrevNodeIsRChildLeaf = False
            
        else: #leaf node
            if PrevNodeIsLChildLeaf:
                inj[id] = inj[id-1]+1
                PrevNodeIsRChildLeaf = True
                PrevNodeIsLChildLeaf = False

            elif PrevNodeIsRChildLeaf:
                num_gens = 0
                par_pos = (inj[id-1]-2)/2
                IsRChild = 1 if par_pos%1 == 0 else 0
                while IsRChild:
                    num_gens+=1
                    par_pos = (par_pos-2)/2
                    IsRChild = 1 if par_pos%1 == 0 else 0
                curr_pos = inj[id-1]
                for i in range(num_gens-1):
                    curr_pos = (curr_pos-2)/2

                inj[id] = int(curr_pos/2)

            else:
               inj[id] = 2*inj[id-1]+1
               PrevNodeIsLChildLeaf = True

    if invert:
        inj = {value:key for key,value in inj.items()}
        #import pdb; pdb.set_trace()
    
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
    func = id_to_pos(tree_clf)
    inv_func = id_to_pos(tree_clf,invert=True)

    padded_tree = add_dummy_nodes(tree_clf)

    true_node_count = 2**(tree.max_depth+1)-1
    num_leaves = sum([1 if tree.feature[i]==-2 else 0 for i in range(tree.node_count)])
    
    assert padded_tree.max_depth == tree.max_depth
    assert padded_tree.node_count == true_node_count
    assert len(padded_tree.threshold) == true_node_count
    assert sum(padded_tree.dummy_nodes) == num_leaves + (true_node_count - tree.node_count)
    for id in range(tree.node_count):
        if tree.feature[id] == -2:
            continue
        assert tree.feature[id] == padded_tree.feature[func(id)]
        assert tree.threshold[id] == padded_tree.threshold[func(id)]
    for pos in range(padded_tree.node_count):
        if padded_tree.dummy_nodes[pos]:
            par_pos = (pos-1)//2
            assert padded_tree.feature[pos] == padded_tree.feature[par_pos]
            assert padded_tree.threshold[pos] == padded_tree.threshold[par_pos]
        else:
            assert padded_tree.feature[pos] == tree.feature[inv_func(pos)]
            assert padded_tree.threshold[pos] == tree.threshold[inv_func(pos)]
