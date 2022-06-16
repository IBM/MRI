import numpy as np
import torch 

######################
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
    
    # @property
    # def loss_fnc(self):
    #     if self.type_ == 'regression':
    #         # if self.Polar_regression:
    #         #     return Polar_MSELoss();                
    #         return torch.nn.MSELoss();             
    #     else: 
    #         return torch.nn.CrossEntropyLoss()

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
        # return torch.tensor([(x_i >= self.bins).sum() for x_i in th]) - 1  # Huh? can't we do this without a for-loop?
        # return (self.continuous_labeller(th).view(-1) >= 0)*1  # Huh: this does not consider self.bins
        return (torch.sin(th).view(-1) >= 0)*1  # Huh: this does not consider self.bins

    
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
        
        
class NullLabeller():
    def __init__(self, num_output):
        self._num_output = num_output
    @property
    def loss_fnc(self):
        return torch.nn.CrossEntropyLoss()

    @property
    def type(self):
        return torch.long #LongTensor
        
    @property
    def num_output(self):
        return self._num_output

    

class Polar_MSELoss(torch.nn.Module):
        
    def helper(self, x_amp, x_angle, target_amp, target_angle):
        angle_cos = (x_angle - target_angle).cos() 
        amp_loss = (x_amp - target_amp)**2/2
        return amp_loss + (1-angle_cos)
    
    def forward(self, x, target):
        assert x.shape[-1] == target.shape[-1] == 2
        x_amp, x_angle = to_polar(x)
        target_amp, target_angle = to_polar(target)
        
        polar_loss = self.helper(x_amp, x_angle, target_amp, target_angle)
        
        mask = x_amp<0.3
        mse_loss = ((x - target)**2).mean(dim=-1)      
        polar_loss[mask] = mse_loss[mask]
        return polar_loss.mean()

# class Polar_MSELoss2(Polar_MSELoss2):

#     def forward(self, x, target):
#         assert x.shape[-1] == target.shape[-1] == 2
#         x_amp, x_angle = to_polar(x)
#         target_amp, target_angle = to_polar(target)
        
#         polar_loss = self.helper(x_amp, x_angle, target_amp, target_angle)
#         polar_loss2 = self.helper(-x_amp, x_angle+np.pi, target_amp, target_angle)
#         polar_loss_min = torch.minimum(polar_loss, polar_loss2)
#         mask = x_amp<0.3
#         mse_loss = ((x - target)**2).mean(dim=-1)
#         polar_loss_min[mask] = mse_loss[mask]
#         return polar_loss_min.mean()
    

def to_polar(input):
    x = input[...,0]
    y = input[...,1]
    return (x**2 + y**2).sqrt(), torch.atan(y/x)+(1+(x>0))*np.pi  
    # complex = (input[:,0] + 1j * input[:,1])
    # return complex.abs(), complex.angle()     #  torch can't differentiate
    # complex_log = complex.log()
    # return complex_log.real, complex_log.imag  #  torch can't differentiate
