import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.nn.utils import clip_grad_norm_
from domainbed.algorithms import ERM, get_optimizer
from domainbed.algorithms import get_algorithm_class as get_algorithm_class_orig
    
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        return get_algorithm_class_orig(algorithm_name)
    else:
        return globals()[algorithm_name]

def get_grad_norm(params):
    norm = 0
    for param in params:
        norm += param.grad.norm()**2
    return norm.sqrt()

def get_weight_norm(params):
    norm = 0
    for param in params:
        norm += param.norm()**2
    return norm.sqrt()

def output_modification(logits, label, loss_type, num_classes):
    if loss_type == 'binary_classification':
        label = label.view(logits.shape).to(logits.dtype) - 0.5
        prob = torch.sigmoid(logits) - 0.5
    elif loss_type == 'classification':
        # if num_classes==2:
        n_env, batch, _ = logits.shape
        logits = (logits[:, :, 1] - logits[:, :, 0]).view(n_env, batch, 1)
        label = label.view(logits.shape).to(logits.dtype) - 0.5
        prob = torch.sigmoid(logits) - 0.5     
    else:
        prob = logits  
    return logits, label, prob


class IRM(ERM):

    constraint_type = 'IRM'
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
        self.optimizer = get_optimizer(self.hparams, self.network.parameters())
        self.update_count = 0
    
    def loss_fn_(self, logits, label_all):
        if len(label_all.shape) == 2:
            label_all=label_all.unsqueeze(-1)
        if len(logits.shape ) == 2:
            logits=logits.unsqueeze(-1)
            
        L0 = self.get_L0(logits, label_all)
        constraint = self.get_constraint(logits, label_all)
        penalty = constraint.abs().pow(2).mean()
        penalty_coeff = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        if self.hparams['loss_type'] in ['regression', 'regression_complex']:
            penalty_coeff *= 4
            
        loss = L0 + penalty_coeff * penalty
        
        return loss, constraint, L0,
    
    def get_L0(self, logits, label_all):
        n_env, batch, n_out = label_all.shape
        logits_ = torch.cat([logits[i,:,:] for i in range(n_env)])
        label_all_ = torch.cat([label_all[i,:,:] for i in range(n_env)])
        if self.hparams['loss_type'] == 'classification':
            label_all_ = label_all_.view(-1)
        return self.loss_fn(logits_, label_all_)
        
    def get_constraint(self, logits, label_all):
        logits, label_all, prob = output_modification(logits, label_all, self.hparams['loss_type'], self.num_classes)     
        oo = (logits.conj()*prob).mean(dim=1)
        oy = (logits.conj()*label_all).mean(dim=1)
               
        constraint_type=self.constraint_type
        if constraint_type=='IRM':
            g = oo - oy
            return g
        else:
            if constraint_type == 'oy':
                g = oy
            elif constraint_type == 'oo-oy':
                g = oo - oy
            else:
                raise ValueError()  
            constraint = g - g.mean() 
            return constraint
        
    def update(self, data_envs, unlabeled=None):        

        input_all = torch.stack([x for x, _ in data_envs])
        label_all = torch.stack([y for _, y in data_envs])
        
        if len(input_all.shape)==5: # for images
            input_all = input_all.view(-1,*input_all.shape[2:])
        if len(label_all.shape) == 2:
            label_all = label_all.unsqueeze(-1)
        
        logits = self.predict(input_all)
        if self.hparams['loss_type'] != 'classification':
            logits = logits.view(label_all.shape)

        loss, constraint, L0 = self.loss_fn_(logits, label_all) 
        logits_var = logits.var(keepdim=True, dim=(1,2)).mean().detach()
        
        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
            self.optimizer = get_optimizer(self.hparams, self.network.parameters())

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(self.network.parameters())
        clip_grad_norm_(self.network.parameters(), self.hparams['grad_clip'])
        self.optimizer.step()
        
        self.update_count += 1

        w_norm = get_weight_norm(self.network.parameters())
        return {'loss': loss.item(), 'nll': L0.item(),  'penalty': constraint.norm().item(), 
                'grad_norm': grad_norm.item(), 'weight_norm': w_norm.item(), 'logits_amp': logits_var.item(),}


class MRI(IRM):
    constraint_type = 'oy'

class IRM_relaxed(IRM):
    constraint_type = 'oo-oy'
        

##########################################################

class IRM_ADMM(ERM):

    constraint_type = 'IRM'
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
        if self.constraint_type == 'IRM':
            num_constraint = num_domains
        else:
            num_constraint = num_domains-1
            self.U_diff = get_ortho_diff(num_domains)
            
        self.network.const_sum = getattr(self.network, 'const_sum', torch.zeros(num_constraint,1))
        # self.network.const_momentum = getattr(self.network, 'const_momentum', torch.zeros(num_constraint,1))
        self.optimizer = get_optimizer(self.hparams, self.network.parameters())
        self.update_count = 0
        self.network.scale = getattr(self.network, 'scale', torch.ones(1,requires_grad=True))

    def loss_fn_(self, logits, label_all): 
        self.network.const_sum = self.network.const_sum.to(logits.device, dtype=logits.dtype)

        if len(label_all.shape) == 2:
            label_all=label_all.unsqueeze(-1)
        if len(logits.shape ) == 2:
            logits=logits.unsqueeze(-1)
        
        L0 = self.get_L0(logits, label_all)
        constraint = self.get_constraint(logits, label_all)
        penalty = (constraint + self.network.const_sum).abs().pow(2).mean() - self.network.const_sum.abs().pow(2).mean()
        penalty_coeff = self.hparams['ADMM_mu']
        if self.hparams['loss_type'] in ['regression', 'regression_complex']:
            penalty_coeff *= 4
            
        loss = L0 + penalty_coeff*penalty/2       
        return loss, constraint.detach(), L0.detach(), 
    
    def get_L0(self, logits, label_all):
        n_env, batch, n_out = label_all.shape
        logits_ = torch.cat([logits[i,:,:] for i in range(n_env)])
        label_all_ = torch.cat([label_all[i,:,:] for i in range(n_env)])
        if self.hparams['loss_type'] == 'classification':
            label_all_ = label_all_.view(-1)
        return self.loss_fn(logits_, label_all_)
    
    def get_constraint(self, logits, label_all):
        logits, label_all, prob = output_modification(logits, label_all, self.hparams['loss_type'], self.num_classes)     
        
        oo = (logits.conj()*prob).mean(dim=1)
        oy = (logits.conj()*label_all).mean(dim=1)
        yy = (label_all.conj()*label_all).mean(dim=1)
               
        constraint_type=self.constraint_type
        if constraint_type=='IRM':
            g = oo - oy
            return g
        else:
            if constraint_type == 'oy':
                g = oy
            elif constraint_type == 'oo-oy':
                g = oo - oy
            else:
                raise ValueError()
            constraint = self.U_diff.to(g.device, dtype=g.dtype) @ g  
            return constraint
        
        
    def update(self, data_envs, unlabeled=None):        

        input_all = torch.stack([x for x, _ in data_envs])
        label_all = torch.stack([y for _, y in data_envs])
                
        if len(input_all.shape)==5: # for images
            input_all = input_all.view(-1,*input_all.shape[2:])
        if len(label_all.shape) == 2:
            label_all = label_all.unsqueeze(-1)
        
        logits = self.predict(input_all)
        if self.hparams['loss_type'] != 'classification':
            logits = logits.view(label_all.shape)
            
        loss, constraint, L0 = self.loss_fn_(logits, label_all)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(self.network.parameters())
        clip_grad_norm_(self.network.parameters(), self.hparams['grad_clip'])
        self.optimizer.step()
        
        self.update_const_sum(constraint)
        self.update_count += 1
        
        scale = self.update_scale(logits, label_all)
        logits_var = scale*logits.var(keepdim=True, dim=(1,2)).mean().detach()
        w_norm = get_weight_norm(self.network.parameters())
        
        return {'loss': loss.item(), 'nll': L0.item(),  'penalty': constraint.norm().item(), 
                'grad_norm': grad_norm.item(), 'weight_norm': w_norm.item(), 'logits_amp': logits_var.abs().item(),
                'Lagrange': self.hparams['ADMM_mu']*self.network.const_sum.norm().item(),
                'scale': scale.abs().item()}
    
    def update_const_sum(self, constraint):
        self.network.const_sum += constraint * self.hparams['ADMM_accum_rate'] * self.hparams['lr']
        
    def update_scale(self, logits, label_all):
        scale = self.network.scale.to(device=logits.device, dtype=logits.dtype)
        loss_scale = self.loss_fn(scale*logits.detach(), label_all)
        scale_grad = autograd.grad(loss_scale, self.network.scale, create_graph=False)
        self.network.scale = self.network.scale - self.hparams['lr_scale_grad']*scale_grad[0]; self.network.scale.grad=None
        return scale


class MRI_ADMM(IRM_ADMM):
    constraint_type = 'oy'

    
class IRM_relaxed_ADMM(IRM_ADMM):
    constraint_type = 'oo-oy'
    
def get_ortho_diff(n_env):
    ones = torch.ones(n_env,1) 
    a,_ = torch.linalg.qr(ones, mode='complete')
    # U_mean = a[0:1]  # ones.T /math.sqrt(n_env)
    U_diff = a[1:]
    return U_diff        