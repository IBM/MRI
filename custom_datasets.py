import os
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision import transforms

from domainbed.datasets import MultipleDomainDataset
from datamodules.sinusoid_datamodule import ShapeTextureGenerator, NWayLabeller, sinusoid


class ShapeTexture(MultipleDomainDataset):
    image_size = (1, 29, 29)
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        
        gen_fnc = sinusoid
        max_phase = hparams['max_phase']
        label_type = hparams['label_type']
        feature_type = hparams['feature_type']
        n_bin = hparams['n_bin']
        total_batch = hparams['total_batch'] or 50000
        env_param_list = hparams['env_param_list'] or self.env_param_list
        causal_param = self.causal_param
        causal_param[self.ENV_variation_type] = hparams['causal_param'] or self.causal_param[self.ENV_variation_type]
        loss_type = hparams['loss_type'] or 'classification'
        
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
        self.input_shape = self.image_size if feature_type == 'images' else (2*n_bin,)

        params_shape   = dict(freq_range=(0.04, 0.0401), gen_fnc=gen_fnc, n_bin=n_bin, max_phase=max_phase)
        params_texture = dict(freq_range=(0.25, 0.251),  gen_fnc=gen_fnc, n_bin=n_bin, max_phase=max_phase)

        environments = self.generate_spurious_envs(env_param_list, self.ENV_variation_type, self.spurious_param_default)
        
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
    
# class ShapeTexture_corr0(ShapeTexture_corr):
#     env_param_list = [0, 0, 0]
#     spurious_param_default = {"offset": 0, "std": 2, "corr": None}

# class ShapeTexture_offset(ShapeTexture):
#     ENV_variation_type = 'offset'
#     env_param_list = [np.pi / 2, -0.4, 0.4]
#     causal_param = {"offset": 0, "std": 0, "corr": 0.8}
#     spurious_param_default = {"offset": None, "std": 0, "corr": 0.9}

# class ShapeTexture_offset2(ShapeTexture):
#     ENV_variation_type = 'offset'
#     env_param_list = [np.pi / 2, -0.2, 0.2]
#     causal_param = {"offset": 0, "std": 0.8, "corr": 1}
#     spurious_param_default = {"offset": None, "std": 0.2, "corr": 1}
    
# class ShapeTexture_offset3(ShapeTexture):
#     ENV_variation_type = 'offset'
#     env_param_list = [np.pi / 2, -0.4, 0.4]
#     causal_param = {"offset": 0, "std": 0.4, "corr": 1}
#     spurious_param_default = {"offset": None, "std": 0.0, "corr": 1}

# class ShapeTexture_offset4(ShapeTexture):
#     ENV_variation_type = 'offset'
#     env_param_list = [np.pi / 2, 0.0, 0.6]
#     causal_param = {"offset": 0, "std": 0.4, "corr": 1}
#     spurious_param_default = {"offset": None, "std": 0.0, "corr": 1}
        
# class ShapeTexture_offset5(ShapeTexture):
#     ENV_variation_type = 'offset'
#     env_param_list = [np.pi / 2, 0.0, 0.4]
#     causal_param = {"offset": 0, "std": 0, "corr": 0.75}
#     spurious_param_default = {"offset": None, "std": 0, "corr": 1}
    
# class ShapeTexture_std(ShapeTexture):
#     ENV_variation_type = 'std'
#     env_param_list = [3.0, 0, 0.6]
#     causal_param = {"offset": 0, "std": 0.7, "corr": 1}
#     spurious_param_default = {"offset": 0, "std": None, "corr": 1}
    
# class ShapeTexture_std2(ShapeTexture_std):
#     causal_param = {"offset": 0, "std": 0.7, "corr": 0.9}
#     spurious_param_default = {"offset": 0, "std": None, "corr": 0.9}
    

class CustomMultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, test_envs, augment, dataset_transform, input_shape, num_classes, total_batch):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        
        self.ENVIRONMENTS = [str(env) for env in environments]

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))[:total_batch] #[::2]
        
        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]
        
        self.datasets = []
        
        augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor()]
        )
        
        for i, env in enumerate(environments):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = None
            self.datasets.append(dataset_transform(images, labels, env, env_transform))
            
        self.input_shape = input_shape
        self.num_classes = num_classes
        
class CustomTensorDataset(Dataset):
    """TensorDataset with support for transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)
    
        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
class CustomColoredMNIST(CustomMultipleEnvironmentMNIST):
    env_param_list = [0.9, 0.1, 0.2]
    causal_param = 0.75

    def __init__(self, root, test_envs, hparams):
        env_param_list = hparams['env_param_list'] or self.env_param_list
        self.causal_param = hparams['causal_param'] or self.causal_param
        augment = hparams['data_augmentation']
        total_batch = hparams['total_batch']
        if hparams['loss_type'] == 'binary_classification':
            num_classes_ = 1
        else:
            num_classes_ = 2
            
        print(self.causal_param, env_param_list,  augment, total_batch)
        super().__init__(root, env_param_list, test_envs, augment, 
                         self.color_dataset, (2, 28, 28,), num_classes_, total_batch)

    def color_dataset(self, images, labels, environment, transform):
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 1-self.causal_param
        labels = torch_xor_(labels,
                                 torch_bernoulli_(1-self.causal_param,
                                                       len(labels)))

        # Assign a color based on the label; flip the color with probability environment
        colors = torch_xor_(labels,
                                 torch_bernoulli_(environment,
                                                       len(labels)))
        
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        
        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return CustomTensorDataset(tensors=(x, y), transform=transform)


class toyCMNIST(MultipleDomainDataset):
    env_param_list = [0.9, 0.1, 0.2]
    causal_param = 0.75
    input_shape = (2,)
    num_classes = 1

    def __init__(self, root, test_envs, hparams):
        super(MultipleDomainDataset, self).__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        
        env_param_list = hparams['env_param_list'] or self.env_param_list
        self.causal_param = hparams['causal_param'] or self.causal_param
        total_batch = hparams['total_batch']
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
            self.datasets.append(
                self.color_dataset(original_labels, env_param_list[i]))

    def color_dataset(self, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).long()
        inputs = (2*labels-1)

        # Flip label with probability 1-self.causal_param
        labels = torch_xor_(labels,
                                 torch_bernoulli_(1-self.causal_param,
                                                       len(labels)))

        # Assign a color based on the label; flip the color with probability environment
        colors = torch_xor_(labels,
                                 torch_bernoulli_(environment,
                                                       len(labels)))
        colors = 2*colors-1
        
        inputs = torch.stack([inputs, colors], dim=1)

        x = inputs.float()
        y = labels.view(-1).long()
        
        return TensorDataset(x, y)


def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()

def torch_xor_(a, b):
    return (a - b).abs()


class CustomMultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        image_shape = 64

        transform = transforms.Compose([
            transforms.Resize((image_shape,image_shape)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(image_shape, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, image_shape, image_shape,)
        self.num_classes = 1 # len(self.datasets[-1].classes)


class CustomTerraIncognita(CustomMultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["test", "train_1", "train_2",]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "custom_terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class CustomVLCS(CustomMultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["test", "train_1", "train_2",]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "custom_VLCS/")
        super().__init__(self.dir, test_envs, hparams["data_augmentation"], hparams)