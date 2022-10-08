import torch 
import numpy as np

def sinusoid(inp):
    return torch.cos(2 * np.pi * inp)

sinusoid.period=np.pi

def Theta_Sampler(th, env, period=np.pi):   # corr: correlation strength (-1 to 1) # width in radian
    
    offset, corr, width = env['offset'], env['corr'], env['std']
    
    th_mean = th + offset
    th_unif = period*torch.rand_like(th_mean)
    th_randn = 1.1*width*(period/2/np.pi)*torch.randn_like(th)

    if corr<0:
        th_randn = th_randn+period/2  # anti-correlated
    
    mask = torch.rand_like(th)<abs(corr)
    th_sample = mask * (th_mean+th_randn) + (~mask)* th_unif
    return th_mean%period, th_sample%period
