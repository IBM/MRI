import numpy as np
import torch 


class NWayLabeller():
    def __init__(self, n_bin=2, mod=np.pi, Polar_regression = False, type_= 'regression'):
        self.type_ = type_
        self.mod = mod           # orientation range (pi for symmetric patterns, 2*pi for asymmetric patterns)
        # self.Polar_regression = Polar_regression

        if type_ == 'classification':
            # assert n_bin >= 2  # Generally
            assert n_bin == 2
        elif type_ == 'binary_classification':
            assert n_bin == 1
        elif type_ == 'regression':
            assert n_bin >= 1
        elif type_ == 'regression_complex':
            assert n_bin == 1

        self.num_output = n_bin

    def __call__(self, th):
        if self.type_ in ['regression', 'regression_complex']:
            return self.continuous_labeller(th)
        elif self.type_ in ['classification', 'binary_classification']:
            return self.discrete_labeller(th % self.mod)
        else:
            raise ValueError()

    @property
    def type(self):
        if self.type_ == 'regression':
            return torch.float #FloatTensor
        elif self.type_ == 'regression_complex':
            return torch.cfloat 
        elif self.type_ == 'binary_classification':
            return torch.float #FloatTensor
        else: 
            return torch.long #LongTensor
        
    def discrete_labeller(self, th):
        th = 2*np.pi*th/self.mod
        return (torch.sin(th).view(-1) >= 0)*1

    
    def continuous_labeller(self, th):
        th = 2*np.pi*th/self.mod
        if self.type_ == 'regression_complex':
            return (torch.cos(th) + 1j*torch.sin(th)).view(-1,1)
        # if self.type_ == 'regression':
        if self.num_output==2:
            return torch.stack([torch.cos(th), torch.sin(th)], dim=1)
        elif self.num_output==1:
            return torch.sin(th).view(-1,1)
        
    def input_labeller(self, th):
        th = 2*np.pi*th/self.mod
        if self.type_ == 'regression_complex':
            return (torch.cos(th) + 1j*torch.sin(th)).view(-1,1)
        else:
            return torch.sin(th).view(-1,1)
