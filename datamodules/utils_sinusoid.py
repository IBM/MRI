import torch 
import numpy as np


def sinusoid(inp):
    return torch.cos(2 * np.pi * inp)

def sawtooth(inp):
    return 2*(inp%1)-1

sinusoid.period=np.pi
sawtooth.period=2*np.pi

######################
class Rotate():
    def __init__(self, offset):# offset in degrees
        self.offset = offset 

    def __call__(self, th):
        return th + self.offset*(np.pi/180)  # convert to radian

class ThetaNoiser():
    def __init__(self, std):
        self.std = std # std in degrees
        
    def __call__(self, th):
        return th + torch.randn_like(th) * (self.std * np.pi/180)



# def Theta_Sampler(th, offset, corr, width=1, period=np.pi):   # corr: correlation strength (-1 to 1) # width in radian
def Theta_Sampler(th, env, period=np.pi):   # corr: correlation strength (-1 to 1) # width in radian
    
    offset, corr, width = env['offset'], env['corr'], env['std']
    
    th_mean = th + offset
    th_unif = period*torch.rand_like(th_mean)
    th_randn = 1.1*width*(period/2/np.pi)*torch.randn_like(th)
    # corr = 2*corr - 1
    if corr<0:
        th_randn = th_randn+period/2  # anti-correlated
    
    mask = torch.rand_like(th)<abs(corr)
    th_sample = mask * (th_mean+th_randn) + (~mask)* th_unif
    return th_mean%period, th_sample%period

# ######################

# def get_angle(x, mod = 2*np.pi):
#     return (x[0] + 1j * x[1] ).detach().log().imag % mod
