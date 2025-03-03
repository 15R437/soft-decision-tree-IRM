import torch.nn as nn
import torch

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