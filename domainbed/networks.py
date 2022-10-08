# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy

from collections import OrderedDict

def get_network(input_shape, num_classes, hparams, factor=1):
    featurizer = Featurizer(input_shape, hparams)
    classifier = Classifier(featurizer.n_outputs*factor, num_classes, hparams)
    return nn.Sequential(OrderedDict([
                        ('featurizer', featurizer),
                        ('classifier', classifier),
                        ]))    

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, n_outputs=None):
        super(Identity, self).__init__()
        self.n_outputs = n_outputs
    def forward(self, x):
        return x

    
class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams.get('resnet18',False):
            self.network = torchvision.models.resnet18(pretrained=True)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(pretrained=True)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x
    
    
class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


class LeNet5(nn.Module):
    n_outputs = 120
    
    def __init__(self, input_shape):
        super().__init__()
        # self.multiplicative_w = nn.Parameter(torch.tensor(-10.))
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1)
        self.average1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.average2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        if input_shape[1] == 64:
            self.fc1 = nn.Linear(12000, self.n_outputs)
        else:
            self.fc1 = nn.Linear(120, self.n_outputs)
    
    def forward(self, xb):
        xb = torch.tanh(self.conv1(xb))
        xb = self.average1(xb)
        xb = torch.tanh(self.conv2(xb))
        xb = self.average2(xb)
        xb = torch.tanh(self.conv3(xb))
        xb = xb.view(xb.shape[0], -1)
        xb = F.relu(self.fc1(xb))
        return xb
    
    
class ST_ConvNet(nn.Module):
    def __init__(self, in_channels=1, 
                 n_channels=None, kernel_size=None, #n_channels=[12,], kernel_size=5,
                 stride=1, pool=2, adapt_pool_size=1):

        super(ST_ConvNet, self).__init__()

        n_channels = n_channels or [12,]
        kernel_size = kernel_size or 5
        kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size]*len(n_channels)
        stride = stride if isinstance(stride, (list, tuple)) else [stride]*len(n_channels)
        self.conv_layers = nn.ModuleList([])

        # instantiate however many conv layers
        for out_channels, k, s in zip(n_channels, kernel_size, stride):
            conv = nn.Conv2d(in_channels, out_channels, k, s)
            self.conv_layers.append(conv)
            in_channels = out_channels

        self.avgpool = nn.AvgPool2d(pool, pool)
        self.adaptavgpool = nn.AdaptiveAvgPool2d(adapt_pool_size)
        self.n_outputs = out_channels * adapt_pool_size ** 2

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = F.relu(conv_layer(x))
            if i < len(self.conv_layers) - 1:
                x = self.avgpool(x)

        x = self.adaptavgpool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x
    

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""

    type = hparams['featurizer_type']
    
    # Below are the default types. for other types, specify hparams['featurizer_type'] for other types. 
    if type is None:
        if len(input_shape) == 1:
            type='identity'
        elif input_shape[1:3] == (28, 28):
            type='MNIST_CNN'
        elif input_shape[1:3] == (29, 29):
            type='ST_ConvNet'
        elif input_shape[1:3] == (32, 32):
            type='Wide_ResNet'
        elif input_shape[1:3] == (224, 224):
            type='ResNet'
        else:
            raise NotImplementedError
    
    Featurizer_dict = get_Featurizer_dict(input_shape, hparams)
    return Featurizer_dict[type]()

def get_Featurizer_dict(input_shape, hparams):
    def _identity():
        return Identity(input_shape[0])
    def _MLP():
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    def _MNIST_CNN():
        return MNIST_CNN(input_shape)
    def _LeNet5():
        return LeNet5(input_shape)
    def _ST_ConvNet():
        if hparams.get("ST_ConvNet_channels"):
            assert isinstance(hparams["ST_ConvNet_channels"], list)
        return ST_ConvNet(n_channels=hparams.get("ST_ConvNet_channels"), 
                          kernel_size=hparams.get("ST_ConvNet_kernel_size"))
    def _Wide_ResNet():
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    def _ResNet():
        return ResNet(input_shape, hparams)

    return {
        'identity': _identity,
        'MLP': _MLP,
        'MNIST_CNN': _MNIST_CNN,
        'LeNet': _LeNet5,
        'ST_ConvNet': _ST_ConvNet,
        'Wide_ResNet': _Wide_ResNet,
        'ResNet': _ResNet
    }


class complexify(nn.Module):
    def forward(self, x):
        return x.to(torch.cfloat)

class complex2real(nn.Module):
    def forward(self, x):
        return torch.cat([x.real, x.imag], dim=-1)
class real2complex(nn.Module):
    def forward(self, x):
        real, imag = x.chunk(2, dim=-1)
        return real + 1j*imag

complexReLU = nn.Sequential( complex2real(),
                             torch.nn.ReLU(),
                             real2complex(),)
                
                
def Classifier(in_features, out_features, hparams):
    type=hparams['classifier_type']

    if type=='identity':
        assert in_features==out_features, 'in_features should be equal to out_features'
        return Identity()
    else:
        if type.startswith('linear'):
            num_layers=type.split('_')[1:]
            num_layers=int(num_layers[0]) if len(num_layers)>0 else 0
            layers = [torch.nn.Linear(in_features, in_features, bias=False) for _ in range(num_layers)]
            layers = layers + [torch.nn.Linear(in_features, out_features, bias=False)]
            
            if hparams['loss_type']=='regression_complex':
                layers = [complexify()] + layers
                return torch.nn.Sequential(*layers).to(torch.cfloat)
            else:
                return torch.nn.Sequential(*layers)

        elif type.startswith('nonlinear_complex'): 
            hidden=type.split('_')[2:]
            hidden=int(hidden[0]) if len(hidden)>0 else 1
            print('hidden =',hidden)
            layers = [ torch.nn.Linear(in_features, hidden, bias=False),
                       complexReLU,
                       torch.nn.Linear(hidden, out_features, bias=False).to(torch.cfloat) ]
            return torch.nn.Sequential(*layers).to(torch.cfloat)

        elif type.startswith('nonlinear_real'): 
            hidden=type.split('_')[2:]
            hidden=int(hidden[0]) if len(hidden)>0 else 1
            print('hidden =',hidden)
            layers = [ torch.nn.Linear(in_features, hidden, bias=False),
                       torch.nn.ReLU(),
                       torch.nn.Linear(hidden, out_features, bias=False)]
            return torch.nn.Sequential(*layers)

        else:
            raise ValueError

class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['classifier_type'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
