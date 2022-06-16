# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import wide_resnet
import copy


from collections import OrderedDict #, defaultdict

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


# Identity = torch.nn.Identity

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

    
class MNIST_CNN_small(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 40*9

    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 20, 3, 2)
        self.conv2 = nn.Conv2d(20, 40, 3, stride=2)
        # self.conv1 = nn.Conv2d(input_shape[0], 16, 5, 2, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=1)

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        # x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

    
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
        
        self.fc1 = nn.Linear(120, self.n_outputs)
        
    # def forward(self, xb):
    #     # xb = xb*torch.stack([self.multiplicative_w.sigmoid(),
    #     #                       1-self.multiplicative_w.sigmoid()])[None,:, None, None]
    #     xb = F.relu(self.conv1(xb))
    #     xb = self.average1(xb)
    #     xb = F.relu(self.conv2(xb))
    #     xb = self.average2(xb)
    #     xb = F.relu(self.conv3(xb))
    #     xb = xb.view(-1, xb.shape[1])
    #     xb = F.relu(self.fc1(xb))
    #     return xb
    
    
    def forward(self, xb):
        xb = F.tanh(self.conv1(xb))
        xb = self.average1(xb)
        xb = F.tanh(self.conv2(xb))
        xb = self.average2(xb)
        xb = F.tanh(self.conv3(xb))
        xb = xb.view(-1, xb.shape[1])
        xb = F.relu(self.fc1(xb))
        return xb
    
    
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

class linear_layer(nn.Module):
    """Just  a single linear layer"""
    def __init__(self, n_inputs, n_outputs):
        super(linear_layer, self).__init__()
        self.output = nn.Linear(n_inputs, n_outputs)
        self.n_outputs = n_outputs
        
    def forward(self, x):
        x = self.output(x)
        return x
    
# class multiplicative_weight(nn.Module):
#     """Just  a single linear layer"""
#     def __init__(self, n_inputs, n_outputs):
#         super().__init__()
#         self.multiplicative_w = nn.Parameter(torch.tensor(-10.))
#         self.n_outputs = n_outputs
        
#     def forward(self, x):
#         if len(x.shape) == 3:
#             x = torch.stack([self.multiplicative_w.sigmoid(), 1 - self.multiplicative_w.sigmoid()]).view(1,1,-1)*x
#         else:
#             x = torch.stack([self.multiplicative_w.sigmoid(), 1 - self.multiplicative_w.sigmoid()]).view(1,-1)*x       
#         return x
    
class ConvNet(nn.Module):
    def __init__(self, in_channels=1, 
                 n_channels=None, kernel_size=None, #n_channels=[12,], kernel_size=5,
                 stride=1, pool=2, adapt_pool_size=1):

        super(ConvNet, self).__init__()

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

# class ConvNet_linear(ConvNet):
#     def __init__(self, n_outputs, in_channels=1, 
#                  n_channels=None, kernel_size=None, #n_channels=[12,], kernel_size=5,
#                  stride=1, pool=2, adapt_pool_size=1):

#         super().__init__(in_channels, n_channels, kernel_size, stride, pool, adapt_pool_size)
#         n_outputs_orig = self.n_outputs
#         self.linear_layer = nn.Linear(n_outputs_orig, n_outputs)
#         self.n_outputs = n_outputs

#     def forward(self, x):
#         x = super().forward(x)
#         x = self.linear_layer(x)
#         return x
    
# class MLP_IRM(nn.Module):
#     n_outputs = 1
    
#     def __init__(self):
#         super(MLP_IRM, self).__init__()
#         self.hidden_dim = 256
#         self.grayscale_model = False
#         if self.grayscale_model:
#             lin1 = nn.Linear(14 * 14, self.hidden_dim)
#         else:
#             lin1 = nn.Linear(2 * 14 * 14, self.hidden_dim)
#         lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         lin3 = nn.Linear(self.hidden_dim, 1)
#         for lin in [lin1, lin2, lin3]:
#             nn.init.xavier_uniform_(lin.weight)
#             nn.init.zeros_(lin.bias)
#         self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
#     def forward(self, input):
#         if self.grayscale_model:
#             out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
#         else:
#             out = input.view(input.shape[0], 2 * 14 * 14)
#         out = self._main(out)
#         return out

def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""

    type = hparams['featurizer_type']
    
    # Below are the default types. for other types, specify hparams['featurizer_type'] for other types. 
    if type is None:
        if len(input_shape) == 1:
            type='identity' #'linear_layer', 'MLP'
        elif input_shape[1:3] == (28, 28):
            type='MNIST_CNN'
        elif input_shape[1:3] == (29, 29):
            type='ConvNet' #if hparams["network_type"] == 'simple' else 'MNIST_CNN'
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
    def _linear_layer():
        return linear_layer(input_shape[0], hparams['featurizer_width'] or input_shape[0])
    def _MLP():
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    def _MNIST_CNN():
        return MNIST_CNN(input_shape)
    def _MNIST_CNN_small():
        return MNIST_CNN_small(input_shape)
    def _LeNet5():
        return LeNet5(input_shape)
    def _ConvNet():
        if hparams.get("convnet_channels"):
            assert isinstance(hparams["convnet_channels"], list)
        return ConvNet(n_channels=hparams.get("convnet_channels"), kernel_size=hparams.get("convnet_kernel_size"))
    # def _ConvNet_linear():
    #     if hparams.get("convnet_channels"):
    #         assert isinstance(hparams["convnet_channels"], list)
    #     return ConvNet_linear(4, n_channels=hparams.get("convnet_channels"), kernel_size=hparams.get("convnet_kernel_size"))
    def _Wide_ResNet():
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    def _ResNet():
        return ResNet(input_shape, hparams)

    return {
        'identity': _identity,
        'linear_layer': _linear_layer,
        'MLP': _MLP,
        'MNIST_CNN': _MNIST_CNN,
        'MNIST_CNN_small': _MNIST_CNN_small,
        'LeNet': _LeNet5,
        'ConvNet': _ConvNet,
        # 'ConvNet_linear': _ConvNet_linear,
        'Wide_ResNet': _Wide_ResNet,
        'ResNet': _ResNet
    }



class masking_layer(nn.Module):
    def __init__(self, n_inputs, n_outputs, init_val=None):
        super().__init__()
        if init_val is None:
            w = 1/4*torch.randn(2)+1/2*torch.rand(2)
        else:
            w = torch.cat([torch.zeros(1)+init_val,torch.ones(1)-init_val])
        self.weight = nn.Parameter(w)
        self.n_outputs = n_outputs
        
    def forward(self, x):
        # x = (1-self.alpha)*x[:,:self.n_outputs] + self.alpha*x[:,self.n_outputs:]
        x = self.weight[1]*x[:,:self.n_outputs] + self.weight[0]*x[:,self.n_outputs:]
        return x

class masking_layer2(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        w = 1/4*torch.randn(4)+1/2*torch.rand(4)
        self.weight = nn.Parameter(w)
        self.n_outputs = n_outputs
        
    def forward(self, x):
        W = torch.stack([self.weight, torch.stack([-self.weight[1], self.weight[0], -self.weight[3], self.weight[2]])])
        x = x @ W.T
        return x

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

    if type=='masking2':  #e.g masking_0, masking_1, masking_0.5
        return masking_layer2(in_features, out_features)
    elif type.startswith('masking'):  #e.g masking_0, masking_1, masking_0.5
        init_val=type.split('_')[1:]
        init_val=float(init_val[0]) if len(init_val)>0 else None
        return masking_layer(in_features, out_features, init_val=init_val)
    elif type=='identity':
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

        # elif type=='nonlinear': 
        #     layers = [ torch.nn.Linear(in_features, in_features // 2),
        #                torch.nn.ReLU(),
        #                torch.nn.Linear(in_features // 2, in_features // 4),
        #                torch.nn.ReLU(),
        #                torch.nn.Linear(in_features // 4, out_features, bias=False)]
        #     return torch.nn.Sequential(*layers)

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
