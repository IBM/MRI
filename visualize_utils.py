import torch 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# def plot_target_vs_out_angle2(target, output, step):
#     fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
#     plt.suptitle(f'Step-{step}', fontsize=15)
#     target=target.cpu()
#     output=output.cpu()
#     if target.dtype==torch.cfloat and target.shape[1]==1 :
#         target=torch.cat([target.real, target.imag], dim=1)
#         output=torch.cat([output.real, output.imag], dim=1)
        
#     ax1.plot(target[:,0],target[:,1],'.', label = 'target')
#     ax1.plot(output[:,0],output[:,1],'.', label = 'output')
#     ax1.legend(loc='upper right')
#     ax1.set_xlabel('output 1', fontsize=15)
#     ax1.set_ylabel('output 2', fontsize=15)
#     ax1.set_ylim(-1.1, 1.1)
#     ax1.set_xlim(-1.1, 1.1)
    
#     target_angle = get_angle2(target.t())
#     output_angle = get_angle2(output.t())
#     ax2.plot(target_angle, output_angle,'.')
#     ax2.set_xlabel('target', fontsize=15)
#     ax2.set_ylabel('output', fontsize=15)
#     ax2.set_ylim(-3.2, 3.2)
#     ax2.set_xlim(-3.2, 3.2)
    
#     # fig = plt.figure(figsize=(9, 4))
#     # plt.title(f'Step-{step}', fontsize=15)
#     # target=target.cpu()
#     # output=output.cpu()
#     # if target.dtype==torch.cfloat and target.shape[1]==1 :
#     #     target=torch.cat([target.real, target.imag], dim=1)
#     #     output=torch.cat([output.real, output.imag], dim=1)
#     # plt.plot(target[:,0], output[:,0],'.')
#     # plt.xlabel('target', fontsize=15)
#     # plt.ylabel('output', fontsize=15)
#     # plt.ylim(-1.1, 1.1)
#     # plt.xlim(-1.1, 1.1)
    
#     # fig.canvas.draw()       # draw the canvas, cache the renderer
#     # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#     # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close()
    
#     return fig

def plot_target_vs_out_angle2(target, output, step):
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
    target_angle = get_angle2(target.t())
    output_angle = get_angle2(output.t())
    ax2.plot(target_angle, output_angle, '.', color='red')
    ax2.tick_params(axis='both', labelsize=15)
    ax2.set_xlabel('Target', fontsize=20)
    ax2.set_ylabel('Output', fontsize=20)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_xlim(-3.2, 3.2)
    plt.close()
    # fig = plt.figure(figsize=(9, 4))
    # plt.title(f'Step-{step}', fontsize=15)
    # target=target.cpu()
    # output=output.cpu()
    # if target.dtype==torch.cfloat and target.shape[1]==1 :
    #     target=torch.cat([target.real, target.imag], dim=1)
    #     output=torch.cat([output.real, output.imag], dim=1)
    # plt.plot(target[:,0], output[:,0],'.')
    # plt.xlabel('target', fontsize=15)
    # plt.ylabel('output', fontsize=15)
    # plt.ylim(-1.1, 1.1)
    # plt.xlim(-1.1, 1.1)
    
    # fig.canvas.draw()       # draw the canvas, cache the renderer
    # image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return fig1, fig2

def get_angle2(x): #, mod = np.pi
    return np.arctan2(x[1], x[0])

######################

def get_angle(x, mod = 2*np.pi):
    return (x[0] + 1j * x[1] ).detach().log().imag % mod

######################
# Visualization

from torch.utils.data import DataLoader
   
def plot_sample_data(dataset_or_loader, num_img = 6):
    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader,  batch_size=num_img, shuffle=False, num_workers=1)
    
    X, Labels = next(iter(loader))  # X, *Ys = next(iter(train_loader))
    Ys = (Labels['shape'], Labels['texture'], Labels['env_idx'])

    plt.figure(figsize=(20, 3.5))
        
    for i, (x, *ys) in enumerate(zip(X,*Ys)):
        ys = [y.data.numpy() for y in ys]   # [y.data.item() for y in ys[:-1]]
        plt.subplot(1, num_img, i+1); 
        ax = plt.imshow(x[0].cpu().numpy())
#         ax.axes.set_title(f'label: {ys}')
        ax.axes.set_title(f'{ys[0][0]:.2f},{ys[0][1]:.2f}')
    plt.show()
        


def plot_target_vs_out_angle(model, task_names, iid_loader, ood_loader):    
    for (loader_type, loader) in (('iid', iid_loader), ('ood', ood_loader)):
        for task_name in task_names:
            title = loader_type + ' ' + task_name
            
            img, targets = next(loader)
            target = targets[task_name].detach()
            out = model(img)[task_name].detach()

            plt.figure(figsize=(9, 4))
            plt.subplot(121);  plt.title(title)
            plt.plot(target[:,0].cpu().numpy(),target[:,1].cpu().numpy(),'.')
            plt.plot(out[:,0].cpu().numpy(),out[:,1].cpu().numpy(),'.')
            
            target_angle = get_angle(target.cpu().t())
            out_angle = get_angle(out.cpu().t())
                
            plt.subplot(122);  plt.title(title); 
            plt.plot(target_angle, out_angle,'.'); 

            plt.show()

        
##################
# from IRM_MAML_v5.visualize_hook import Hook

# def concat_to_one_img(tensor):
#     if len(tensor.shape)==4:
#         tensor = torch.cat(tensor.split(1, dim=0),dim=-2).squeeze(0)
#         tensor = torch.cat(tensor.split(1, dim=0),dim=-1).squeeze(0)
#     return tensor

        
# def get_conv_activations(model, layer, image):
#     """
#     takes a model, a module (specific layer in a met), and an image, and returns the
#     activations at that layer using the Hook class.
#     """
#     hook = Hook(layer)
#     model(image)
#     activations = hook.output.squeeze(0).detach()
#     hook.close()
# #     return activations.clip(min=0)  # ReLU layer
#     return activations

# def visualize_conv_layers_with_hook(model, iid_img, ood_img):

#     fc_weight = model.fc.weight
#     layers = [layer for layer in model.conv_layers] + [model.fc]
    
# #     angle = get_angle(fc_weight)
# #     val, idx = angle.sort()
    
# #     plt.figure(figsize=(13, 4))
# #     plt.subplot(131);  plt.title('empty')
# #     plt.subplot(132);  plt.imshow(concat_to_one_img(iid_img.transpose(0,1))); plt.title('iid input'); plt.xlabel('img #'); #plt.ylabel('in channel')
# #     plt.subplot(133);  plt.imshow(concat_to_one_img(ood_img.transpose(0,1))); plt.title('ood input'); plt.xlabel('img #'); #plt.ylabel('in channel')
# #     plt.show()
        
#     cmap = None
# #     cmap = plt.get_cmap('coolwarm')
# #     cmap = plt.get_cmap('seismic')
    
#     for layer in layers:
#         w = layer.weight.detach()#[idx]
        
# #         if len(w.shape)==4:
# # #             plt.figure(figsize=(1+3*2*w.shape[1], 1+2*w.shape[0]))
# # #             plt.figure(figsize=(13, 4))
# #             plt.figure(figsize=(13, 2*int(1+0.5*w.shape[0])))
# #         else:
# #             plt.figure(figsize=(6, 1*int(1+0.1*w.shape[0])))
        
# #         w = concat_to_one_img(w).squeeze(0)
# #         plt.subplot(131);  plt.imshow(w, cmap=cmap);   plt.title('weight'); plt.xlabel('in channel'); plt.ylabel('out channel')

# #         activity = get_conv_activations(model, layer, iid_img)#[idx]
# #         activity = concat_to_one_img(activity.transpose(0,1))
# #         plt.subplot(132);     plt.imshow(activity);   plt.title('iid activity'); plt.xlabel('img #'); plt.ylabel('out channel')

# #         activity = get_conv_activations(model, layer, ood_img)#[idx]
# #         activity = concat_to_one_img(activity.transpose(0,1))
# #         plt.subplot(133);     plt.imshow(activity);   plt.title('ood activity'); plt.xlabel('img #'); plt.ylabel('out channel')
# #         plt.show()

        
#         #

#         if len(w.shape)==4:
# #             plt.figure(figsize=(1+3*2*w.shape[1], 1+2*w.shape[0]))
# #             plt.figure(figsize=(13, 4))
#             plt.figure(figsize=(3*int(1+0.5*w.shape[0]),2*int(1+0.5*w.shape[1])))
# #             plt.figure(figsize=(int(1+0.5*w.shape[0]),13))
#         else:
# #             plt.figure(figsize=(13, int(1+0.1*w.shape[0])))
# #             plt.figure(figsize=(int(1+0.1*w.shape[0]),13))
#             plt.figure(figsize=(1*int(1+0.5*w.shape[0]),int(1+0.5*w.shape[1])))
        
#         w = concat_to_one_img(w).squeeze(0)
#         plt.subplot(131);  plt.imshow(w.t(), cmap=cmap);   plt.title('weight'); plt.ylabel('in channel'); plt.xlabel('out channel')

#         activity = get_conv_activations(model, layer, iid_img)#[idx]
#         activity = concat_to_one_img(activity.transpose(0,1))
#         plt.subplot(132);     plt.imshow(activity.t());   plt.title('iid activity'); plt.ylabel('img #'); plt.xlabel('out channel')

#         activity = get_conv_activations(model, layer, ood_img)#[idx]
#         activity = concat_to_one_img(activity.transpose(0,1))
#         plt.subplot(133);     plt.imshow(activity.t());   plt.title('ood activity'); plt.ylabel('img #'); plt.xlabel('out channel')
#         plt.show()
        

def visualize_conv_layers_with_hook(model, iid_img, ood_img): #

    fc_weight = model.classifier[-1].weight
    layers = [layer for layer in model.featurizer.conv_layers] + [model.classifier[-1]]
    
    cmap = None
    
    for layer in layers:
        
        # plot weights
        w = layer.weight.detach()#[idx]

        if len(w.shape)==4:
            fig, axs = plt.subplots(3, 1, figsize=(16, 32)) #
            w = concat_to_one_img(w).squeeze(0)
        else:
            fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        
        axs[0].imshow(w.t(), cmap=cmap)
        axs[0].set_title('weight')
        axs[0].set_ylabel('in channel')
        axs[0].set_xlabel('out channel')

        # plot iid activations
        activity = get_conv_activations(model, layer, iid_img)#[idx]
        activity = concat_to_one_img(activity.transpose(0,1))
        axs[1].imshow(activity.t());   axs[1].set_title('iid activity'); axs[1].set_ylabel('img #'); axs[1].set_xlabel('out channel')

         # plot ood activations
        activity = get_conv_activations(model, layer, ood_img)#[idx]
        activity = concat_to_one_img(activity.transpose(0,1))
        axs[2].imshow(activity.t());   axs[2].set_title('ood activity'); axs[2].set_ylabel('img #'); axs[2].set_xlabel('out channel')
        plt.show()
        
    # plot iid and ood images
    fig, axs = plt.subplots(1, 2, figsize=(8, 10))
    iid_plot = concat_to_one_img(iid_img.transpose(0,1))
    axs[0].imshow(iid_plot.t())
    axs[0].set_title('iid activity'); axs[0].set_ylabel('img #')
    ood_img = concat_to_one_img(ood_img.transpose(0,1))
    axs[1].imshow(ood_img.t())
    axs[1].set_title('ood activity'); axs[1].set_ylabel('img #')
    plt.show()