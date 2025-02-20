import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from sklearn.tree import DecisionTreeClassifier
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils import SoftTreeArgs

#Deterministic Function
NUM_ENVS = 10
rand_X = [torch.randn(100,4) for i in range(NUM_ENVS)]
rand_y = [torch.Tensor([1 if (x[0]>0.5 and x[1]<0.5 and x[3]<0.5) or 
                    (x[0]<0.5 and x[2]>0.5)else 0 for x in X]) for X in rand_X]
X = torch.stack(rand_X)
y = torch.stack(rand_y).long()
envs = [DataLoader(TensorDataset(X[i,:,:],y[i,:]),batch_size=20) for i in range(NUM_ENVS)]
erm_dataloader = DataLoader(TensorDataset(torch.cat(rand_X,dim=0),torch.cat(rand_y,dim=0).long()),batch_size=200)

test_X = torch.randn(100,4)
test_y = torch.Tensor([1 if (x[0]>0.5 and x[1]<0.5 and x[3]<0.5) or 
                    (x[0]<0.5 and x[2]>0.5)else 0 for x in test_X])
test_loader = DataLoader(TensorDataset(test_X,test_y.long()))
args = SoftTreeArgs(input_dim=4,output_dim=2)

NUM_EPOCHS = 100
NUM_TRIALS = 1
erm_test_accuracy = []
irm_test_accuracy = []

for trial in range(NUM_TRIALS):
    soft_tree_erm = SoftDecisionTree(args)
    soft_tree_irm = SoftDecisionTree(args)
    print(f"ERM Training {trial+1}")
    for epoch in range(1,NUM_EPOCHS+1):
        soft_tree_erm.train_erm(erm_dataloader,epoch)
    print(f"IRM Training {trial+1}")
    for epoch in range(1,NUM_EPOCHS+1):
        soft_tree_irm.train_irm(envs,epoch,penalty_anneal_iters=50,l1_weight=0.01,max_one_weight=0.1)
    
    erm_test_accuracy.append(soft_tree_erm.test_(test_loader,print_result=False,return_acc=True))
    irm_test_accuracy.append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))

X_ = torch.cat(rand_X,dim=0).detach().numpy()
y_ = torch.cat(rand_y,dim=0).detach().numpy()
hard_tree = DecisionTreeClassifier(max_depth=args.max_depth)
hard_tree.fit(X_,y_)
preds = torch.tensor(hard_tree.predict(X_))

print(f"hard tree acc: {100.*torch.mean((torch.cat(rand_y,dim=0)==preds).float()):0.2f}%")
print(f"soft tree (erm) acc: {torch.mean(torch.tensor(erm_test_accuracy)):0.2f}%")
print(f"soft tree (irm) acc: {torch.mean(torch.tensor(irm_test_accuracy)):0.2f}%")