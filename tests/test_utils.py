from src.utils import add_dummy_nodes, decision_tree_penalty
from src.model import SoftDecisionTree
from sklearn.tree import DecisionTreeClassifier
import pytest
import numpy as np

def id_to_pos(tree_clf):
    """
    This returns an injective function from the ordering
    of the sklearn tree object to our preferred ordering
    on a padded tree
    """
    tree = tree_clf.tree_
    inj = {0:0} #the root node always has index 0 in both orderings
    for id in range(1,tree.node_count):
        pass

    def func(id:int):
        return inj[id]
    
    return func

rand_X = [np.random.randn(100,4) for i in range(10)]
rand_y = [np.array([1. if x[0]>0 else 0. for x in X]) for X in rand_X]
rand_trees = [DecisionTreeClassifier(max_depth=2).fit(X,y) for X,y in zip(rand_X,rand_y)]
@pytest.mark.parametrize("tree_clf",rand_trees)
def test_add_dummy_nodes(tree_clf):
    tree = tree_clf.tree_
    true_node_count = 2**(tree.max_dept+1)-1
    padded_tree = add_dummy_nodes(tree_clf)

    assert padded_tree.max_depth == tree.max_dept
    assert padded_tree.node_count == true_node_count
    assert sum(padded_tree.dummy_nodes) == true_node_count - tree.node_count
    for node in range(true_node_count):
        assert padded_tree.feature == 2
        assert padded_tree.threshold == 3
