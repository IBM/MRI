import torch 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_target_vs_output_angle(target, output, step):
    target=target.cpu()
    output=output.cpu()
    if target.dtype==torch.cfloat and target.shape[1]==1 :
        target=torch.cat([target.real, target.imag], dim=1)
        output=torch.cat([output.real, output.imag], dim=1)
        
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax1.plot(target[:,0],target[:,1],'.', label = 'target', color='blue')
    ax1.plot(output[:,0],output[:,1],'.', label = 'output', color='red')
    ax1.tick_params(axis='both', labelsize=15)
    ax1.set_xlabel('Real', fontsize=20)
    ax1.set_ylabel('Imaginary', fontsize=20)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlim(-1.3, 1.3)
    ax1.legend(['Target', 'Output'], bbox_to_anchor=(0.45, 1.15), markerscale=3.,
               loc='upper center', ncol=3, fontsize=20, frameon=False)
    plt.close()
    
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    target_angle = get_angle(target.t())
    output_angle = get_angle(output.t())
    ax2.plot(target_angle, output_angle, '.', color='red')
    ax2.tick_params(axis='both', labelsize=15)
    ax2.set_xlabel('Target', fontsize=20)
    ax2.set_ylabel('Output', fontsize=20)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_xlim(-3.2, 3.2)
    plt.close()
    
    return fig1, fig2

def get_angle(x):
    return np.arctan2(x[1], x[0])