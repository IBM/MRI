import torch
import torch.nn.functional as F
from domainbed.datasets import MultipleDomainDataset, ColoredMNIST, CustomTensorDataset
from datamodules.sinusoid_datamodule_v2 import ShapeTextureGenerator, NWayLabeller, sinusoid, sawtooth
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST

class ShapeTexture(MultipleDomainDataset):
    # input_shape_choices = {'images': (1, 29, 29), 'factors': (4,), 'factored_nonlinear': (2,)}
    image_size = (1, 29, 29)
    def __init__(self, root, test_envs, hparams):
        super().__init__()

#         normalize_transform = torchvision.transforms.Normalize((0.5), (0.5))
        
        gen_fnc = sinusoid
        total_batch = hparams.get('total_batch', 100000)
        max_phase = hparams.get('max_phase', 0)
        label_type = hparams.get('label_type', 'shape')
        feature_type = hparams.get('feature_type', 'images')
        n_bin = hparams.get('n_bin', 2)
        
        env_param_list = hparams.get('env_param_list', self.env_param_list)
        
        causal_param = self.causal_param
        causal_param[self.ENV_variation_type] = hparams.get('causal_param', self.causal_param[self.ENV_variation_type])

        loss_type = hparams.get('loss_type', 'classification')
        
        if loss_type == 'classification':
            assert n_bin >= 2, 'n_bin should be >=2 for classification'
        if loss_type == 'binary_classification':
            assert n_bin == 1, 'n_bin should be 1 for binary_classification'
        else:
            assert n_bin in [1,2], 'n_bin should be 1,2 for regression'
       
        label_fncs = dict(shape=NWayLabeller(n_bin=n_bin, mod=gen_fnc.period, type_ = loss_type),
                          texture=NWayLabeller(n_bin=n_bin, mod=gen_fnc.period, type_ = loss_type))
        
        self.ENVIRONMENTS = [str(env) for env in env_param_list]
        self.num_classes  = label_fncs[label_type].num_output
        
        # assert feature_type in self.input_shape_choices.keys(), 'feature type must be one of images, factors or factored_nonlinear'
        self.input_shape = self.image_size if feature_type == 'images' else (2*n_bin,)

        params_shape   = dict(freq_range=(0.04, 0.0401), gen_fnc=gen_fnc, n_bin=n_bin, max_phase=max_phase)
        params_texture = dict(freq_range=(0.25, 0.251),  gen_fnc=gen_fnc, n_bin=n_bin, max_phase=max_phase)

        environments = self.generate_spurious_envs(env_param_list, self.ENV_variation_type, self.spurious_param_default)
        # print('causal_env_factor:', self.causal_param)
        # print('spurious_env_factors:', environments)
        
        def dataset_helper(shape_env, texture_env):
            generator = ShapeTextureGenerator(params_shape, params_texture,
                                              shape_env, texture_env,
                                              feature_type=feature_type,
                                              image_size=self.image_size[-1])

            x, y_shape, y_texture = generator(label_fncs, total_batch=total_batch)
            if label_type == 'shape':
                y = y_shape.type(label_fncs['shape'].type)
            else:
                y = y_texture.type(label_fncs['texture'].type)
                
            # x = normalize_transform(x)
            # if len(y.shape) == 1 and loss_type == 'regression':
            #     y = y.reshape(-1,1)
            # print(x.shape, y.shape)
            return TensorDataset(x, y)
        
        if label_type == 'shape':
            self.datasets = [ dataset_helper(shape_env=causal_param, texture_env=spurious_param) 
                             for spurious_param in environments]
        else:
            self.datasets = [ dataset_helper(shape_env=spurious_param, texture_env=causal_param) 
                             for spurious_param in environments]
            
    def generate_spurious_envs(self, env_params, keyword, default_params):
        return [{**default_params, keyword: param} for param in env_params]
    
import numpy as np
class ShapeTexture_corr(ShapeTexture):
    ENV_variation_type = 'corr'
    env_param_list = [0, 1.0, 0.8]
    causal_param = {"offset": 0, "std": 0, "corr": 0.75}
    spurious_param_default = {"offset": 0, "std": 0, "corr": None}
    
class ShapeTexture_corr0(ShapeTexture_corr):
    env_param_list = [0, 0, 0]
    spurious_param_default = {"offset": 0, "std": 2, "corr": None}

class ShapeTexture_offset1(ShapeTexture):
    ENV_variation_type = 'offset'
    env_param_list = [np.pi / 2, -0.4, 0.4]
    causal_param = {"offset": 0, "std": 0, "corr": 0.8}
    spurious_param_default = {"offset": None, "std": 0, "corr": 0.9}

class ShapeTexture_offset2(ShapeTexture):
    ENV_variation_type = 'offset'
    env_param_list = [np.pi / 2, -0.2, 0.2]
    causal_param = {"offset": 0, "std": 0.8, "corr": 1}
    spurious_param_default = {"offset": None, "std": 0.2, "corr": 1}
    
class ShapeTexture_offset3(ShapeTexture):
    ENV_variation_type = 'offset'
    env_param_list = [np.pi / 2, -0.4, 0.4]
    causal_param = {"offset": 0, "std": 0.4, "corr": 1}
    spurious_param_default = {"offset": None, "std": 0.0, "corr": 1}

class ShapeTexture_offset4(ShapeTexture):
    ENV_variation_type = 'offset'
    env_param_list = [np.pi / 2, 0.0, 0.6]
    causal_param = {"offset": 0, "std": 0.4, "corr": 1}
    spurious_param_default = {"offset": None, "std": 0.0, "corr": 1}
        
class ShapeTexture_offset5(ShapeTexture):
    ENV_variation_type = 'offset'
    env_param_list = [np.pi / 2, 0.0, 0.4]
    causal_param = {"offset": 0, "std": 0, "corr": 0.75}
    spurious_param_default = {"offset": None, "std": 0, "corr": 1}
    
class ShapeTexture_std(ShapeTexture):
    ENV_variation_type = 'std'
    env_param_list = [3.0, 0, 0.6]
    causal_param = {"offset": 0, "std": 0.7, "corr": 1}
    spurious_param_default = {"offset": 0, "std": None, "corr": 1}
    
class ShapeTexture_std2(ShapeTexture_std):
    causal_param = {"offset": 0, "std": 0.7, "corr": 0.9}
    spurious_param_default = {"offset": 0, "std": None, "corr": 0.9}
    
    
    
class FactoredCMNIST_mixed(MultipleDomainDataset):
    # ENVIRONMENTS = ['+90%', '+80%', '-90%']
    env_param_list = [0.9, 0.1, 0.2]
    causal_param = 0.75
    input_shape = (4,)
    num_classes = 1

    def __init__(self, root, test_envs, hparams):
        super(MultipleDomainDataset, self).__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        
        env_param_list = hparams.get('env_param_list', self.env_param_list)
        self.causal_param = hparams.get('causal_param', self.causal_param)
        total_batch = hparams.get('total_batch', None)
        if hparams['loss_type'] == 'classification': self.num_classes = 2
            
        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_labels))[:total_batch]

        original_labels = original_labels[shuffle]

        self.datasets = []
        
        self.ENVIRONMENTS = [str(env) for env in env_param_list]
        
        for i in range(len(env_param_list)):
            labels = original_labels #[i::len(env_param_list)]
            self.datasets.append(
                self.color_dataset(labels, env_param_list[i]))

    def color_dataset(self, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).long()
        images = F.one_hot(labels, num_classes=2)
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(1-self.causal_param,
                                                       len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :] *= 0

        x = images.view(len(images),-1).float()
        y = labels.view(-1).long()
        
        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class FactoredCMNIST_demixed(FactoredCMNIST_mixed):
    input_shape = (2,)

    def color_dataset(self, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).long()
        images = (2*labels-1)

        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(1-self.causal_param,
                                                       len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        colors = 2*colors-1
        
        images = torch.stack([images, colors], dim=1)

        x = images.float()
        y = labels.view(-1).long()
        
        return TensorDataset(x, y)
    
    
class ColoredMNISTv2(ColoredMNIST):

    def color_dataset(self, images, labels, environment, transform):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(1-self.causal_param,
                                                       len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        colors = 2*colors-1
        
        # Create a new channel that only contains color information
        images = torch.stack([images, images], dim=1)
        images[:,1,:,:] = colors[:, None, None].byte()
        
        x = images.float().div_(255.0)
        y = labels.view(-1).long()
        
        return CustomTensorDataset(tensors=(x, y), transform=transform)

