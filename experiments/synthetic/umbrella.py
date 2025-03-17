import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
from src.model import SoftDecisionTree
from utils.general import SoftTreeArgs
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from generate_data import generate_and_save,func_stochastic,func_sigmoid
from utils.general import FeatureMask, decision_tree_penalty

#LOADING DATA
curr_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(curr_dir,"data/umbrella_data.pickle")
if os.path.exists(file_path):
    print("loading existing data..")
    with open(file_path,'rb') as file:
        train_data,test_data,best_params = pickle.load(file)
else:
    print(f"generating new data..")
    train_envs = [0.1,0.2,0.3] #if we bring an umbrella, we are likely to also bring a raincoat
    test_envs = [0.9] # if we bring an umbrella, we are unlikely to also bring a raincoat
    param_grid = {
        'penalty_anneal_iters': [95],
        'penalty_weight': [1],
        'l1_weight_feat': [100],
        'l1_weight_tree': [10],
        'lr':[0.1],
        'num_epochs':[100],
        'lmbda': [0.1],
        'depth_discount_factor': [1]}

    train_data,test_data,best_params = generate_and_save(n_train_samples=10000,n_test_samples=1000,train_envs=train_envs,test_envs=test_envs,
          y_func=func_stochastic,param_grid=param_grid,save_as="data/umbrella_data.pickle",batch_size=1000,random_seed=0)

train_data_irm = train_data['irm_envs']
train_data_erm = train_data['erm_loader']
X_train_raw,y_train_raw = train_data['raw_data']

test_loader = test_data['erm_loader']
X_test_raw,y_test_raw = test_data['raw_data']

best_lr = best_params['lr'] #0.1
best_l1_feat = best_params['l1_weight_feat'] #100
best_l1_tree = best_params['l1_weight_tree'] #10
best_penalty_weight = best_params['penalty_weight'] #1
best_penalty_anneal = best_params['penalty_anneal_iters'] #95
tree_args = SoftTreeArgs(input_dim=3,output_dim=2,batch_size=1000,lr=best_lr,max_depth=3,log_interval=1)

#EXPERIMENT 1: soft tree (irm) vs soft tree (erm) vs hard tree vs random forest test accuracy
def experiment_1(num_trials):
    print("Running Experiment 1..")
    hard_tree,random_forest = DecisionTreeClassifier(max_depth=tree_args.max_depth,random_state=0), RandomForestClassifier(n_estimators=10,random_state=0)
    hard_tree.fit(X_train_raw,y_train_raw)
    random_forest.fit(X_train_raw,y_train_raw)

    tree_preds,forest_preds = hard_tree.predict(X_test_raw),random_forest.predict(X_test_raw)
    tree_acc, forest_acc = accuracy_score(y_test_raw,tree_preds),accuracy_score(y_test_raw,forest_preds)

    erm_test_accuracy = []
    irm_test_accuracy = []

    for trial in range(num_trials):
        soft_tree_erm = SoftDecisionTree(tree_args)
        soft_tree_irm = SoftDecisionTree(tree_args)
        print(f"ERM Training {trial+1}")
        for epoch in range(1,51):
            soft_tree_erm.train_erm(train_data_erm,epoch,l1_weight_tree=best_l1_tree)
        print(f"IRM Training {trial+1}")
        for epoch in range(1,101):
            soft_tree_irm.train_irm(train_data_irm,epoch,penalty_anneal_iters=best_penalty_anneal,l1_weight_feat=best_l1_feat,
                                    l1_weight_tree=best_l1_tree,penalty_weight=best_penalty_weight)
        
        erm_test_accuracy.append(soft_tree_erm.test_(test_loader,print_result=False,return_acc=True))
        irm_test_accuracy.append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))



    print(f"hard tree acc: {100*tree_acc:0.2f}%")
    print(f"random forest acc: {100*forest_acc:.2f}%")
    print(f"soft tree (erm) acc: {np.mean(erm_test_accuracy):0.2f}%")
    print(f"soft tree (irm) acc: {np.mean(irm_test_accuracy):0.2f}%")

    results_file_path = os.path.join(curr_dir,'results/umbrella-experiment-1.pickle')

    models = ['hard tree', 'random forest', 'soft tree (erm)', 'soft tree (irm)']
    acc = [100*tree_acc,100*forest_acc,np.mean(erm_test_accuracy),np.mean(irm_test_accuracy)]
    with open(results_file_path,"wb") as file:
        pickle.dump({'models':models,'accuracy':acc,'hard_tree':hard_tree},file)

    plt.bar(models,acc)
    plt.title("Umbrella Use Prediction")
    plt.xlabel("model")
    plt.ylabel("test accuracy (%)")

    plot_file_path = os.path.join(curr_dir,'plots/umbrella-experiment-1')
    plt.savefig(plot_file_path)
    plt.show()


#EXPERIMENT 2: initialising feature mask weights and seeing how this affects learning and final accuracy
def experiment_2(num_trials,init_weights:list,ideal_weight=np.array([1,1,0]),num_epochs=100):
    """
    Takes a list (init_weights) of 1D tensors of size input_dim. These are passed as the initial
    weights of a feature mask for a soft decision tree which is then trained for num_epochs epochs.
    The softmax-negative-cross-entropy loss between the learned weights and ideal_weights is
    calculated after each epoch. After num_epochs, the test accuracy is also calculated.
    """

    print("Running Experiment 2..")

    irm_test_accuracy = []
    weight_loss = []
    figure = plt.figure()
    num_plots = len(init_weights)
    for i,w in enumerate(init_weights):
        phi = FeatureMask(input_dim=3,init_weight=nn.Parameter(w))
        tree_args = SoftTreeArgs(input_dim=3,output_dim=2,batch_size=1000,lr=best_lr,max_depth=3,log_interval=1,phi=phi)

        acc = []
        loss = [[] for _ in range(num_epochs)]
        for trial in range(num_trials):
            soft_tree_irm = SoftDecisionTree(tree_args)
            print(f"IRM Training {trial+1}")
            for epoch in range(1,num_epochs+1):
                soft_tree_irm.train_irm(train_data_irm,epoch,penalty_anneal_iters=best_penalty_anneal,l1_weight_feat=best_l1_feat,
                                        l1_weight_tree=best_l1_tree,penalty_weight=best_penalty_weight)
                learned_weight = soft_tree_irm.phi.weight.detach().cpu().numpy()

                #computing softmax and then negative cross-entropy
                learned_prob = np.exp(learned_weight-np.max(learned_weight))/np.exp(learned_weight-np.max(learned_weight)).sum()
                ideal_prob = np.exp(ideal_weight-np.max(ideal_weight))/np.exp(ideal_weight-np.max(ideal_weight)).sum()
                nce = -np.sum(learned_prob*np.log(ideal_prob))

                loss[epoch-1].append(nce)
           
            acc.append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))

        weight_loss.append(np.mean(loss,axis=1))
        irm_test_accuracy.append(np.mean(acc))

        #plotting epochs vs weight loss
        figure.add_subplot(1,num_plots,i+1)
        plt.plot([1.*i for i in range(1,num_epochs+1)],np.mean(loss,axis=1))
        plt.xlabel('epoch')
        plt.ylabel('feature weight loss (nce)')
    
    results_file_path = os.path.join(curr_dir,'results/umbrella-experiment-1.pickle')
    with open(results_file_path,"wb") as file:
        pickle.dump({'weight_loss':weight_loss,'irm_test_accuracy':irm_test_accuracy},file)

    plot_file_path = os.path.join(curr_dir,'plots/umbrella-experiment-2')
    plt.savefig(plot_file_path)
    plt.show()


#EXPERIMENT 3: graphing test accuracy against penalty_anneal_iters
def experiment_3(num_trials,anneal_list=[i for i in range(0,110,10)],num_epochs=100):
    print("Running Experiment 3..")
    irm_test_accuracy = [[] for _ in anneal_list]
    for trial in range(num_trials):
        print(f"IRM Training {trial+1}")
        for id,anneal in enumerate(anneal_list):
            soft_tree_irm = SoftDecisionTree(tree_args)
            for epoch in range(1,num_epochs+1):
                soft_tree_irm.train_irm(train_data_irm,epoch,penalty_anneal_iters=anneal,l1_weight_feat=best_l1_feat,
                                        l1_weight_tree=best_l1_tree,penalty_weight=best_penalty_weight,print_progress=False)
                
            irm_test_accuracy[id].append(soft_tree_irm.test_(test_loader,print_result=False,return_acc=True))
    
    mean_accuracy = np.mean(irm_test_accuracy,axis=1)
    results_file_path = os.path.join(curr_dir,'results/umbrella-experiment-3.pickle')
    with open(results_file_path,"wb") as file:
        pickle.dump({'penalty_anneal_iters':anneal_list,'mean_accuracy':mean_accuracy},file)

    plt.plot(anneal_list,mean_accuracy)
    plt.xlabel("penalty_anneal_iters")
    plt.ylabel("mean test accuracy")
    plt.title(f"Test accuracy versus penalty_anneal_iters over {num_epochs}")

    plot_file_path = os.path.join(curr_dir,'plots/umbrella-experiment-3')
    plt.savefig(plot_file_path)
    plt.show()
    return mean_accuracy

#EXPERIMENT 4: Fix the weights of the soft tree to their ideal weights. Let phi = (1,1,c) and vary c from 0 to 1. For each value of c,
#fit a hard tree and compute the decision tree penalty between this hard tree and the soft tree over phi
def experiment_4(c_list):
    init_tree_weights = {}
    for i in range(15):
        pass
    penalty = []
    for c in c_list:
        phi = FeatureMask(input_dim=3,init_weight=nn.Parameter(torch.tensor([1.,1.,c*1.])))
        tree_args = SoftTreeArgs(input_dim=3,output_dim=2,batch_size=1000,lr=best_lr,max_depth=3,log_interval=1,phi=phi,
                                 init_weights=init_tree_weights)
        soft_tree = SoftDecisionTree(tree_args)
        penalty.append(decision_tree_penalty(soft_tree,X_train_raw,y_train_raw).item())
    
    plt.plot(c_list,penalty)
    plt.xlabel('c')
    plt.ylabel('penalty')

    plot_file_path = os.path.join(curr_dir,'plots/umbrella-experiment-4')
    plt.savefig(plot_file_path)
    plt.show()

    plt.show()

#RUN EXPERIMENTS HERE
#experiment_1(10)
#experiment_2(10,init_weights=[torch.tensor([.5,.5,.5]),torch.tensor([0.,0.,0.]),torch.tensor([1.,1.,1.])])
#experiment_3(10)
experiment_4(c_list=np.linspace(0,1,11))

#PLOT GRAPHS HERE
