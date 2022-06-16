# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import Dataset, TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

# from datamodules.sinusoid_datamodule_v2 import ShapeTextureGenerator, NWayLabeller, sinusoid, sawtooth

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    # "Debug28",
    # "Debug224",
    # Linear
    # "FactoredCMNIST",
    # Shape Texture
    # "ShapeTexture",
    # "ShapeTexture_corr",
    # "ShapeTexture_corr0",
    # "ShapeTexture_offset1",
    # "ShapeTexture_offset2",
    # "ShapeTexture_std",
    # Small images
    "ColoredMNIST",
    # "CSColoredMNIST",
    "CSMultiColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    # "SVIRO",
    # # WILDS datasets
    # "WILDSCamelyon",
    # "WILDSFMoW"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
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
    """TensorDataset with support of transforms.
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
    

class ColoredMNIST(MultipleEnvironmentMNIST):
#     ENVIRONMENTS = ['+90%', '+80%', '-90%']
    env_param_list = [0.9, 0.1, 0.2]

    def __init__(self, root, test_envs, hparams):
        self.causal_param = hparams.get('causal_param', 0.75)
        total_batch = hparams.get('total_batch', None)
        env_param_list = hparams.get('env_param_list', self.env_param_list)
        augment = hparams.get('data_augmentation', False)
        if hparams['loss_type'] == 'binary_classification':
            num_classes_ = 1
        else:
            num_classes_ = 2
        super().__init__(root, env_param_list, test_envs, augment, 
                         self.color_dataset, (2, 28, 28,), num_classes_, total_batch)

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
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0
        
        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return CustomTensorDataset(tensors=(x, y), transform=transform)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()
    
    
# class CSColoredMNIST(MultipleEnvironmentMNIST):
# #     ENVIRONMENTS = ['+90%', '+80%', '-90%']
#     env_param_list = [0.9, 0.1, 0.2]

#     def __init__(self, root, test_envs, hparams):
#         env_param_list = hparams.get('env_param_list', self.env_param_list)
#         augment = hparams.get('data_augmentation', None) 
#         super().__init__(root, env_param_list, test_envs, augment, self.color_dataset, (2, 28, 28,), 2)

#     def color_dataset(self, images, labels, environment, transform):
#         # # Subsample 2x for computational convenience
#         # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
#         # Assign a binary label based on the digit
#         labels = (labels < 5).float()

#         colors = self.torch_bernoulli_(
#             0.5, len(labels)).float()  # sample color for each sample

#         w_comb = 1 - self.torch_xor_(labels, colors)  # compute xor of label and
#         # color and negate it

#         selection_0 = torch.nonzero(w_comb == 0, as_tuple=True)[0]  # indices
#         # where -xor is zero
#         selection_1 = torch.nonzero(w_comb == 1, as_tuple=True)[0]  # indices
#         # were -xor is one

#         ns0 = len(selection_0)
#         ns1 = len(selection_1)

#         final_selection_0 = selection_0[
#             torch.nonzero(self.torch_bernoulli_(environment, ns0) == 1,
#                           as_tuple=True)[0]]  # -xor =0 then select that point
#         # with probability prob_e

#         final_selection_1 = selection_1[
#             torch.nonzero(self.torch_bernoulli_(1 - environment, ns1) == 1,
#                           as_tuple=True)[0]] # -xor =0 then select that point
#         # with probability 1-prob_e

#         final_selection = torch.cat((final_selection_0, final_selection_1))
#         # indices of the final set of points selected

#         colors = colors[final_selection]  # colors of the final set
#         # of selected points
#         labels = labels[final_selection]  # labels of the final set of selected
#         # points
#         images = images[final_selection]  # gray scale image of the
#         # final set of selected points

#         images = torch.stack([images, images], dim=1)
#         # Apply the color to the image by zeroing out the other color channel
#         images[torch.tensor(range(len(images))), (
#             1 - colors).long(), :, :] *= 0

#         x = images.float().div_(255.0)
#         y = labels.view(-1).long()

#         return CustomTensorDataset(tensors=(x, y), transform=transform)

#     def torch_bernoulli_(self, p, size):
#         return (torch.rand(size) < p).float()

#     def torch_xor_(self, a, b):
#         return (a - b).abs()


# class CSMultiColoredMNIST(MultipleEnvironmentMNIST):
# #     ENVIRONMENTS = ['+100%', '+90%', '0%']
#     env_param_list = [0.0, 1.0, 0.9]

#     def __init__(self, root, test_envs, hparams):
#         env_param_list = hparams.get('env_param_list', self.env_param_list)
#         augment = hparams.get('data_augmentation', None) 
#         self.env_seed = 1
#         self.colors = torch.FloatTensor(
#             [[0, 100, 0], [188, 143, 143], [255, 0, 0], [255, 215, 0],
#              [0, 255, 0], [65, 105, 225], [0, 225, 225],
#              [0, 0, 255], [255, 20, 147], [160, 160, 160]])
#         self.random_colors = torch.randint(255, (10, 3)).float()
        
#         super().__init__(root, env_param_list, test_envs, augment, self.color_dataset, (3, 28, 28,), 10)


#     def color_dataset(self, images, labels, environment, transform):
#         original_seed = torch.cuda.initial_seed()
#         torch.manual_seed(self.env_seed)
#         shuffle = torch.randperm(len(self.colors))
#         self.colors_ = self.colors[shuffle]
#         torch.manual_seed(self.env_seed)
#         ber = self.torch_bernoulli_(environment, len(labels))
#         # print("ber:", len(ber), sum(ber))
#         torch.manual_seed(original_seed)

#         images = torch.stack([images, images, images], dim=1)
#         # binarize the images
#         images = (images > 0).float()
#         y = labels.view(-1).long()
#         color_label = torch.zeros_like(y).long()

#         # Apply the color to the image
#         for img_idx in range(len(images)):
#             if ber[img_idx] > 0:
#                 color_label[img_idx] = labels[img_idx]
#                 for channels in range(3):
#                     images[img_idx, channels, :, :] = images[img_idx, channels, :,
#                                                       :] * \
#                                                       self.colors_[labels[
#                                                                        img_idx].long(), channels]
#             else:
#                 color = torch.randint(10, [1])[
#                     0]  # random color, regardless of label
#                 color_label[img_idx] = color
#                 for channels in range(3):
#                     images[img_idx, channels, :, :] = images[img_idx, channels, :,
#                                                       :] * self.colors_[
#                                                           color, channels]

#         x = images.float().div_(255.0)

#         return CustomTensorDataset(tensors=(x, y), transform=transform)

#     def torch_bernoulli_(self, p, size):
#         return (torch.rand(size) < p).float()

#     def torch_xor_(self, a, b):
#         return (a - b).abs()



class RotatedMNIST(MultipleEnvironmentMNIST):
    env_param_list = [0, 15, 30, 45, 60, 75]

    def __init__(self, root, test_envs, hparams):
        total_batch = hparams.get('total_batch', None)
        env_param_list = hparams['env_param_list'] or self.env_param_list
        augment = False
        super().__init__(root, env_param_list, test_envs, augment,
                         self.rotate_dataset, (1, 28, 28,), 10, total_batch)

    def rotate_dataset(self, images, labels, angle, transform):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
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

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

