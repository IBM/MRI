import numpy as np

import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .labeller import NWayLabeller, NullLabeller
from .utils_sinusoid import sinusoid, sawtooth, Theta_Sampler #Apply_Offset #Rotate, ThetaNoiser

from core.decorators import continuous_loader


"""
A Torchvision Dataset and Dataloader for the Sinusoid based shape vs texture dataset.

Generally, the images are composed of a low frequency sinusoid and a high frequency sinusoid multiplied together, yielding a low frequency (shape) pattern overlayed with a high frequency (texture) pattern. The objective is to classify (or regress) the orientation (theta) of the low (or high) frequency component. Correlations can be introduced between the high and low frequency components. When different correlations are used in the train vs test set, we can then study out of distribution generalization.

A typical set up might be: 

Train distributions:
d1 ∼ θ2 = θ1 + 0
d2 ∼ θ2 = θ1 + 30
d3 ∼ θ2 = θ1 + 60

Test distributions:
d4 ∼ θ2 = θ1 + 90

Where θ1 and θ2 are the orientation of the low and high frequency components respectively.
"""

# normalize_transform = transforms.Compose([ transforms.Normalize((0.5), (0.5)) ])


######################
class ShapeTextureDataset(Dataset):
    """
    Shape vs Texture Dataset class
    consumes a list of instantiated Generator objects and generates a dataset of specified length.
    """
    def __init__(self, generators, label_fncs, transform=None, total_batch=50000): #generators:LIST(ShapeTextureGenerator)
        self.transform = transform or transforms.Compose([])
        # total_batch = total_batch // len(generators)  # commented out to make ERM and IRM loaders consistent 

        xs=[]
        y1s=[]
        y2s=[]
        env_ids = []

        for generator_i, generator in enumerate(generators): 
            x, y1, y2 = generator(label_fncs, total_batch)
            xs.append(x)
            y1s.append(y1.type(label_fncs['shape'].type))
            y2s.append(y2.type(label_fncs['texture'].type))
            env_ids.append((generator_i*torch.ones(len(x))).type(label_fncs['env_idx'].type))

        self.x = torch.cat(xs)
        self.y1 = torch.cat(y1s)
        self.y2 = torch.cat(y2s)
        self.env_id = torch.cat(env_ids)

#         assert len(self.x) == len(self.y1) == len(self.y2) == len(self.env_id)


    def __getitem__(self, index):
        x, y1, y2, env_id = self.x[index], self.y1[index], self.y2[index], self.env_id[index]
        x = self.transform(x)
        labels = dict(shape=y1, texture=y2, env_idx=env_id)
        return x, labels
    
    def __len__(self):
        return self.x.shape[0]

######################
class ShapeTextureGenerator():
    """ 
    Generates Sinusoid stimuli with specified statistics
    """
    def __init__(self, params_shape, params_texture, shape_env, texture_env, image_size=28, background_noise = 0, feature_type='images', random = True): #, **kwargs):

        self.params_shape = params_shape  # parameters for shape feature  (low freq)
        self.params_texture = params_texture  # parameters for texture feature (high freq)
        
        self.shape_env = shape_env
        self.texture_env = texture_env
        
        self.image_size = image_size
        self.background_noise = background_noise
        
        # assert feature_type in ['images', 'factors', 'factored_nonlinear']
        self.feature_type = feature_type
        self.random = random

        
    def __call__(self, label_fncs, total_batch, seed=None, random=True):
        if seed:
            np.random.seed(seed)

        th_range =  (0, self.params_shape['gen_fnc'].period)  # (0,2*np.pi)  
        if self.random:
            th0 = torch.FloatTensor(total_batch).uniform_(*th_range)
        else:
            th0 = torch.linspace(th_range[0], th_range[1], total_batch)
        
        # generate shape params -- invariant from environment to environment
        freq1, ph1 = get_env_params(self.params_shape, self.shape_env, th0)
        # generate texture params -- not invariant from environment to environment
        freq2, ph2 = get_env_params(self.params_texture, self.texture_env, th0)
        
        ## Old setting
        # th1_mean, th1_sampled = Theta_Sampler(th0, self.shape_env, self.params_shape['gen_fnc'].period)
        # th2_mean, th2_sampled = Theta_Sampler(th1_mean, self.texture_env, self.params_texture['gen_fnc'].period)
        # th_shape, th_texture, th_label = th1_sampled, th2_sampled, th1_mean

        ## New setting
        th1_mean, th1_sampled = Theta_Sampler(th0, self.shape_env, self.params_shape['gen_fnc'].period)
        th2_mean, th2_sampled = Theta_Sampler(th1_sampled, self.texture_env, self.params_texture['gen_fnc'].period)
        th_shape, th_texture, th_label = th0, th2_sampled, th1_sampled
        
        y1 = label_fncs['shape'](th_label)
        y2 = label_fncs['texture'](th2_mean)   #not used for now
        
        if self.feature_type == 'images':
            x = sinusoid_image(
                fnc1 = self.params_shape['gen_fnc'], freq1=freq1, th1=th_shape, ph1=ph1, 
                fnc2 = self.params_texture['gen_fnc'], freq2=freq2, th2=th_texture, ph2=ph2, 
                image_size=self.image_size, background_noise=self.background_noise ).unsqueeze(1)
        elif self.feature_type == 'factors':
            x1 = label_fncs['shape'].input_labeller(th_shape)
            x2 = label_fncs['texture'].input_labeller(th_texture)
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = torch.stack([th_shape, th_texture], dim=-1) 
#             x = torch.stack([th1_mean, th2_mean], dim=-1) 
            
        return x, y1, y2    
    
        
def get_env_params(params, env, th0): 
    # th_mean, th_sample = Theta_Sampler(th0, env, params['gen_fnc'].period)
    freq = torch.FloatTensor(th0.shape).uniform_(*params['freq_range'] )
    ph = torch.FloatTensor(th0.shape).uniform_(0, params['max_phase']) #(0, 1)  # random phase
    return freq, ph

    
######################
def sinusoid_image(fnc1, freq1, th1, ph1, fnc2, freq2, th2, ph2, image_size, background_noise):
    """generates sinusoid images with specified params -- 
    images are [1, image_size, image_size] with pixels between 0 and 1
    """
    def get_image_gen(fnc, th, freq, ph):
        freq_x = freq*torch.cos(th);    freq_y = freq*torch.sin(th)
        def image_gen(coord_x, coord_y):
            input = freq_x.view(-1,1,1) * coord_x + freq_y.view(-1,1,1) * coord_y + ph.view(-1,1,1)
            return (fnc(input) + 1)/2
        return image_gen   

    w = image_size // 2
    grid_val = torch.arange(-w, w+1, dtype=torch.float)

    image_gen1 = get_image_gen(fnc1, th1, freq1, ph1)
    image_gen2 = get_image_gen(fnc2, th2, freq2, ph2)

    coord_x, coord_y = torch.meshgrid(grid_val, grid_val)
    coord_x, coord_y = coord_x.unsqueeze(0), coord_y.unsqueeze(0)

    s1 = image_gen1(coord_x, coord_y)
    s2 = image_gen2(coord_x, coord_y)
    
    out =  s1*s2 

    # out *= (1+background_noise*torch.randn_like(out))  # multiplicative background noise
    out += background_noise*torch.randn_like(out)        # additive background noise
    return out
#     return out.bernoulli()  # stochastic version




######################

def get_ShapeTextureDataLoaders(shape_env, texture_envs_dict, total_batch, batch_size=64, feature_type='images', gen_fncs = (sinusoid, sinusoid), n_bins=(0,0), shuffle=True, num_workers=1, **kwargs): 
    # can we use more workers? 
    
    """    takes in a texture_envs_dict like 
        {   'train' : [[-15, 0, 15]],
            'iid_test' : [[-15, 0, 15]],
            'ood_test' : [[90]],        }

    and returns a dictionary with lists of loaders like
        {   'train' : [list of loaders],
            'iid_test' : [list of loaders],
            'ood_test' : [list of loaders],        }
    """


    label_fncs = dict(shape = NWayLabeller(n_bin=n_bins[0], mod=gen_fncs[0].period),  
                      texture = NWayLabeller(n_bin=n_bins[1], mod=gen_fncs[1].period),
                      env_idx = NullLabeller(len(texture_envs_dict['train'])) )  # env_idx is ugly ?
    
    shape_freq_range=(0.04, 0.0401)  # (0.02, 0.03)
    texture_freq_range=(0.25, 0.251)  # (0.175, 0.2)
    max_phase=0
#     max_phase=1
    params_shape   = dict(freq_range=shape_freq_range, gen_fnc=gen_fncs[0], n_bin=n_bins[0], max_phase=max_phase) #, period=gen_fncs[0].period)#   **default_params } 
    params_texture = dict(freq_range=texture_freq_range, gen_fnc=gen_fncs[1], n_bin=n_bins[1], max_phase=max_phase) #, period=gen_fncs[1].period)#   **default_params } 

    def helper_fnc(texture_env_list):  # get_ShapeTextureDataLoader
        generators = [ ShapeTextureGenerator(params_shape, params_texture, 
                                             shape_env = shape_env, texture_env=texture_env, 
                                             feature_type=feature_type, **kwargs) 
                      for texture_env in texture_env_list]
    
        dataset = ShapeTextureDataset(generators=generators, label_fncs=label_fncs, total_batch=total_batch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return continuous_loader(loader) # if continuous_loading else loader

    loaders = {}
    for key, texture_envs in texture_envs_dict.items():  # key = 'train', 'test' ....
        loaders[key] = [helper_fnc(texture_env)  for texture_env in texture_envs  ]
    
    return loaders, label_fncs 


def get_params_envs(iid_list, ood_list, keyword='corr', default_params = {"offset": 0, "std": 0, "corr": 0}):

    params_envs = {
        'train' :  [[ {**default_params, keyword:param}] for param in iid_list], # three separate train loaders (environments)
        'iid_test' : [[ {**default_params, keyword:param} for param in iid_list]], # three separate train loaders (environments)
        'ood_test' :[[ {**default_params, keyword:param} for param in ood_list]]
    }
    return params_envs
