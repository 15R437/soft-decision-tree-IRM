import os
import time

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils.general import decision_tree_penalty,new_tree_penalty

"""class UpperTriangularWeight(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.rand(size, size)) #initialie a matrix with entries in [0,1)
        self.register_buffer("mask", torch.triu(torch.ones_like(self.weight))) #a mask for the upper triangle bit

    def forward(self):
        return self.weight * self.mask
    
    def set_weight_as(self,w):
        #error handling - make sure size of w is consistent with size
        self.weight=nn.Parameter(w)"""

def assign_init_value(pos,init_val=None):
    if init_val!= None:
        try:
            assert init_val[pos]!=None
            return init_val[pos]
        except:
            pass
    return None

class InnerNode():

    def __init__(self, depth, args, pos):
        self.args = args
        self.pos = pos
        self.fc = assign_init_value(pos,self.args.tree_weights)
        if self.fc == None:
            self.fc = nn.Linear(self.args.input_dim, 1)
        self.fc =self.fc.to(self.args.device)
        beta = assign_init_value(pos,self.args.beta)
        if beta == None:
            beta = torch.randn(1).to(self.args.device)
        self.beta = nn.Parameter(beta).to(self.args.device)
        self.leaf = False
        self.prob = None
        self.leaf_accumulator = []
        self.lmbda = self.args.lmbda * 2 ** (-depth)
        self.build_child(depth)
        self.penalties = []

    def reset(self):
        self.leaf_accumulator = []
        self.penalties = []
        self.left.reset()
        self.right.reset()

    def build_child(self, depth):
        if depth < self.args.max_depth:
            self.left = InnerNode(depth+1, self.args,2*self.pos+1)
            self.right = InnerNode(depth+1, self.args,2*self.pos+2)
        else :
            self.left = LeafNode(self.args,2*self.pos+1)
            self.right = LeafNode(self.args,2*self.pos+2)

    def forward(self, x):
        return(F.sigmoid(self.beta*self.fc(x)))
    
    def select_next(self, x):
        prob = self.forward(x)
        if prob < 0.5:
            return(self.left, prob)
        else:
            return(self.right, prob)

    def cal_prob(self, x, path_prob):
        self.prob = self.forward(x) #probability of selecting right node
        self.path_prob = path_prob
        left_leaf_accumulator = self.left.cal_prob(x, path_prob * (1-self.prob))
        right_leaf_accumulator = self.right.cal_prob(x, path_prob * self.prob)
        self.leaf_accumulator.extend(left_leaf_accumulator)
        self.leaf_accumulator.extend(right_leaf_accumulator)
        return(self.leaf_accumulator)

    def get_penalty(self):
        penalty = (torch.sum(self.prob * self.path_prob) / torch.sum(self.path_prob), self.lmbda)
        if not self.left.leaf:
            left_penalty = self.left.get_penalty()
            right_penalty = self.right.get_penalty()
            self.penalties.append(penalty)
            self.penalties.extend(left_penalty)
            self.penalties.extend(right_penalty)
        return(self.penalties)


class LeafNode():
    def __init__(self, args, pos):
        self.args = args
        param = assign_init_value(pos,self.args.leaf_dist)
        if param == None:
            param = torch.randn(self.args.output_dim)
        
        self.param = nn.Parameter(param).to(self.args.device)
        self.leaf = True
        self.softmax = nn.Softmax(dim=1)

    def forward(self):
        return(self.softmax(self.param.view(1,-1)))

    def reset(self):
        pass

    def cal_prob(self, x, path_prob):
        Q = self.forward()
        #Q = Q.expand((self.args.batch_size, self.args.output_dim))
        Q = Q.expand((path_prob.size()[0], self.args.output_dim))
        return([[path_prob, Q]])


class SoftDecisionTree(nn.Module):

    def __init__(self, args):
        super(SoftDecisionTree, self).__init__()
        self.args = args
        self.root = InnerNode(1, self.args,0)
        self.phi = self.args.phi.to(self.args.device)
        
        self.collect_parameters() ##collect parameters and modules under root node
        #self.optimizer = optim.Adam(self.parameters(),lr=self.args.lr)
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.test_acc = []
        self.define_extras(self.args.batch_size)
        self.best_accuracy = 0.0

    def define_extras(self, batch_size):
        ##define target_onehot and path_prob_init batch size, because these need to be defined according to batch size, which can differ
        self.target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
        self.target_onehot = Variable(self.target_onehot).to(self.args.device)
        self.path_prob_init = Variable(torch.ones(batch_size, 1)).to(self.args.device)
    '''
    def forward(self, x):
        node = self.root
        path_prob = Variable(torch.ones(self.args.batch_size, 1))
        while not node.leaf:
            node, prob = node.select_next(x)
            path_prob *= prob
        return node()
    '''    
    def cal_loss(self, x, y,include_featuriser=True):
        batch_size = y.size()[0]
        if include_featuriser: 
            x = self.phi(x)
        
        leaf_accumulator = self.root.cal_prob(x, self.path_prob_init)
        loss = 0.
        max_prob = [-1. for _ in range(batch_size)]
        max_Q = [torch.zeros(self.args.output_dim) for _ in range(batch_size)]
        for (path_prob, Q) in leaf_accumulator:
            TQ = torch.bmm(y.clone().view(batch_size, 1, self.args.output_dim).to(Q.dtype), torch.log(Q).clone().view(batch_size, self.args.output_dim, 1)).view(-1,1)
            loss += path_prob * TQ
            path_prob_numpy = path_prob.cpu().data.numpy().reshape(-1)
            for i in range(batch_size):
                if max_prob[i] < path_prob_numpy[i]:
                    max_prob[i] = path_prob_numpy[i]
                    max_Q[i] = Q[i]
        loss = loss.mean()
        penalties = self.root.get_penalty()
        C = 0.
        for (penalty, lmbda) in penalties:
            C -= lmbda * 0.5 *(torch.log(penalty) + torch.log(1-penalty))
        output = torch.stack(max_Q)
        self.root.reset() ##reset all stacked calculation
        return(-loss, output, C) ## -log(loss) will always output non, because loss is always below zero. I suspect this is the mistake of the paper?

    def collect_parameters(self,include_featuriser=True):
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        if include_featuriser: 
            self.param_list.extend([param.data for param in self.phi.parameters()])
            #self.module_list.extend(self.phi.layers) instead, I access self.phi.layers directly for the l1_loss
        while nodes:
            node = nodes.pop(0)
            if node.leaf:
                param = node.param
                self.param_list.append(param)
            else:
                fc = node.fc
                beta = node.beta
                nodes.append(node.left) #change to append node.left first and then node.right after 
                nodes.append(node.right)
                self.param_list.append(beta)
                self.module_list.append(fc)

    def train_erm(self, train_loader, epoch, print_progress=True,return_stats=False,l1_weight_tree=0):
        self.train()
        self.define_extras(self.args.batch_size)
        for batch_idx, (data, target) in enumerate(train_loader):
            correct = 0
            data, target = data.to(self.args.device), target.to(self.args.device)
            #data = data.view(self.args.batch_size,-1)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            self.optimizer.zero_grad()
            
            loss, output,C = self.cal_loss(data, self.target_onehot,include_featuriser=False)
            loss = loss+C
            l1_loss = torch.tensor(0.).to(self.args.device)
            for fc in self.module_list:
                for w in fc.weight:
                    l1_loss += torch.norm(w,p=1)

            loss += l1_weight_tree*l1_loss
            #loss /= l1_weight_tree
            #loss.backward(retain_variables=True)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            accuracy = 100. * correct / len(data)

            if print_progress:
                if batch_idx % self.args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item(), #changed loss.data[0] to loss.data.item()
                        correct, len(data),
                        accuracy))
            if return_stats:
                return {'loss':loss, 'acc':accuracy}
    
    def train_irm(self,envs,epoch,print_progress=True,return_stats=False,
                  penalty_weight=100,penalty_anneal_iters=50,depth_discount_factor=2,
                  l1_weight_feat=10,l1_weight_tree=10,max_one_weight=0,phi_clip_val=None,grad_clip_val=None):
        """
        We expect envs to be a list of data loaders, each one corresponding to a different environment.
        One training loop involves taking the first batch from each environment (dataloader), computing losses, updating 
        our parmaeters and then moving on to the next set of batches until we exhaust the batches.

        We assume that each environment has the same number of batches.
        """
        self.train()
        self.define_extras(self.args.batch_size)
        NUM_BATCHES = min([len(e) for e in envs])
        num_envs = len(envs)
        start_time = time.time()
        total_loss = 0.
        for batch_idx in range(1,NUM_BATCHES+1):
            batch_data = []
            batch_target = []
            penalty = torch.tensor(0.).to(self.args.device)
            for e in envs:
                data,target = next(iter(e))
                batch_data.append(data)
                batch_target.append(target)
                if epoch > penalty_anneal_iters:
                    target_ = target.view(-1,1).to(self.args.device)
                    data = data.to(self.args.device)
                    batch_size = target_.size()[0]
                    """if not batch_size == self.args.batch_size:
                        self.define_extras(batch_size)
                    self.target_onehot.data.zero_()            
                    self.target_onehot.scatter_(1, target_, 1.)"""
                    target_onehot = torch.FloatTensor(batch_size, self.args.output_dim)
                    target_onehot = Variable(target_onehot).to(self.args.device)
                    target_onehot.data.zero_()
                    target_onehot.scatter_(1,target_,1.)
                    data = self.phi(data)
                    #import pdb; pdb.set_trace()
                    penalty += new_tree_penalty(self,data,target)

            data = torch.cat(batch_data).to(self.args.device)
            target = torch.cat(batch_target).to(self.args.device)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            data = Variable(data)
            if not batch_size == self.args.batch_size:
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)

            train_loss,output,C = self.cal_loss(data,self.target_onehot)
            total_loss+= train_loss
            train_loss+= C            
            #featuriser regularisation
            l1_loss_feat = torch.tensor(0.).to(self.args.device)
            for module in self.phi.layers:
                try: 
                    module.weight
                except: 
                    continue
                for w in module.weight:
                    l1_loss_feat += torch.norm(w,p=1)

            #regularisation for soft tree weights
            l1_loss_tree = torch.tensor(0.).to(self.args.device)
            for fc in self.module_list:
                for w in fc.weight:
                    l1_loss_tree += torch.norm(w,p=1)

            avg_penalty = penalty_weight*penalty/(num_envs)

            loss = train_loss + avg_penalty + l1_weight_feat*l1_loss_feat + l1_weight_tree*l1_loss_tree

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            grad_norm = 0.
            with torch.no_grad():
                grad_norm += nn.utils.clip_grad_norm_(self.parameters(),float('inf'),norm_type=2)
                if phi_clip_val!=None:
                    list(self.phi.parameters())[-1].data.clamp_(0,phi_clip_val)
                if grad_clip_val!=None:
                    nn.utils.clip_grad_norm_(self.parameters(),grad_clip_val)

            pred = output.data.max(1)[1].to(self.args.device) # get the index of the max log-probability
            correct = (pred==target).cpu().sum()
            accuracy = 100. * correct / batch_size
            
            
            if print_progress:
                if batch_idx % self.args.log_interval == 0:
                    end_time = time.time()
                    num_data_processed = batch_idx*num_envs*batch_size
                    total_data = num_envs*batch_size*NUM_BATCHES
                    #formatted_accuracy = "%, ".join(list(map(lambda acc: f"{acc.item():.2f}",accuracy)))
                    print(f"""Train Epoch: {epoch} [{num_data_processed}/{total_data} ({100.*batch_idx/NUM_BATCHES:.0f}%)]\t Train Loss: {train_loss.data.item():.4f}, Avg Penalty: {avg_penalty.data.item():.4f}, L1 Feat Loss: {(l1_weight_feat*l1_loss_feat).data.item():.2f}, L1 Tree Loss: {(l1_weight_tree*l1_loss_tree).data.item():.2f}, Accuracy: {correct.sum().item()}/{batch_size*num_envs} ({100.*correct.sum().item()/(batch_size*num_envs):.2f})% Time Elapsed:{end_time-start_time:.2f}s, Grad Norm: {grad_norm:.2f}""")
                    start_time = time.time()
        if return_stats:
            return {'loss':total_loss/NUM_BATCHES, 'acc':accuracy}
        end_time = time.time()

    def test_(self, test_loader,print_result=True,return_stats=False):
        self.eval()
        self.define_extras(self.args.batch_size)
        test_loss = 0.
        correct = 0
        for data, target in test_loader:
            data, target = data.to(self.args.device), target.to(self.args.device)
            target = Variable(target)
            target_ = target.view(-1,1)
            batch_size = target_.size()[0]
            data = data.view(batch_size,-1)
            ##convert int target to one-hot vector
            data = Variable(data)
            if not batch_size == self.args.batch_size: #because we have to initialize parameters for batch_size, tensor not matches with batch size cannot be trained
                self.define_extras(batch_size)
            self.target_onehot.data.zero_()            
            self.target_onehot.scatter_(1, target_, 1.)
            loss, output,C = self.cal_loss(data, self.target_onehot)
            test_loss+= loss
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy = 100. * correct / len(test_loader.dataset)
        if print_result:
            print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
            correct, len(test_loader.dataset),
            accuracy))
        self.test_acc.append(accuracy)

        if accuracy > self.best_accuracy:
            #self.save_best('./result')
            self.best_accuracy = accuracy
        if return_stats:
            return {'loss':test_loss/len(test_loader),'acc':accuracy}

    def save_best(self, path):
        try:
            os.makedirs('./result')
        except:
            print('directory ./result already exists')

        with open(os.path.join(path, 'best_model.pkl'), 'wb') as output_file:
            pickle.dump(self, output_file)
