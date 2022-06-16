import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
# from torch.autograd import Variable

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

# def add_grad_penalty(loss, params, penalty_coeff):
#     grads = autograd.grad(loss, params, create_graph=True)
#     grad_norm = torch.stack([grad.norm()**2 for grad in grads]).sum().sqrt()
#     loss = loss + penalty_coeff*grad_norm
#     return loss

def get_weight_norm(params):
    norm = 0
    for param in params:
        norm += param.norm()**2
    return norm.sqrt()

# def set_imag_to_zero(params):
#     for param in params:
#         with torch.no_grad():
#             if param.dtype == torch.cfloat:
#                 param.imag -= param.imag

def output_modification(logits, label, loss_type, num_classes):
    if loss_type == 'binary_classification':
        label = label.view(logits.shape).to(logits.dtype) - 0.5
        prob = torch.sigmoid(logits) - 0.5
    elif loss_type == 'classification':
        if num_classes==2:
            n_env, batch, _ = logits.shape
            logits = (logits[:, :, 1] - logits[:, :, 0]).view(n_env, batch, 1)
            label = label.view(logits.shape).to(logits.dtype) - 0.5
            prob = torch.sigmoid(logits) - 0.5         
        # else: #if self.num_classes>=2:
        #     label = torch.nn.functional.one_hot(label, num_classes=logits.shape[-1]).to(logits.dtype)
        #     prob = nn.Softmax(dim=1)(logits)            
    else:
        prob = logits   
        # logits *= 2     
    return logits, label, prob
    
# class rev_IRM(ERM):
#     """Invariant Risk Minimization"""
    
#     def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
#         super().__init__( input_shape, num_classes, num_domains, hparams, **kwargs)
#         self.register_buffer('update_count', torch.tensor([0]))
#         self.grad_clip =  hparams['grad_clip']

#     def _irm_grad(self, logits, label):
#         # if self.dummy:
#         if self.hparams['loss_type'] == 'binary_classification':
#             label = label.view(logits.shape).to(logits.dtype) - 0.5
#         dummy = torch.tensor(1.).to(logits.device).requires_grad_()
#         params = [dummy]
#         label = label*dummy
#         # else:
#         #     params = self.network.classifier.parameters()
            
#         loss = self.loss_fn(logits, label)
#         grads = autograd.grad(loss, params, create_graph=True)
#         grads = torch.cat([g.view(-1) for g in grads])
#         return grads, loss    
                
#     def compute_loss(self, data_envs):
#         nll = 0.
#         penalty = 0.
#         grads = []
#         logits_all = []
        
#         for i, (x, y) in enumerate(data_envs):
#             logits = self.network(x)
#             grad, loss = self._irm_grad(logits, y)
#             nll += loss
#             grads.append(grad)
#             logits_all.append(logits)
        
#         grads = torch.stack(grads) 
#         logits_var = torch.stack(logits_all).var(keepdim=True, dim=(1,2)).mean().detach()
        
#         penalty = ((grads - grads.mean(keepdim=True, dim=0)).abs()**2).sum()
        
#         return nll/len(data_envs), penalty/len(data_envs), logits_var 

    
#     def compute_loss_with_penalty(self, loss, penalty, logits_amp):
#         penalty_weight = self.hparams['irm_lambda']  if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 1
        
#         # if self.adaptive_penalty:
#         #     penalty_weight *= logits_amp
            
#         if self.hparams['irm_type'] in [None, 'additive']:
#             loss = loss + penalty_weight * penalty
#         elif self.hparams['irm_type'] == 'multiplicative':
#             loss = loss * (1 + penalty_weight * penalty)
#         elif self.hparams['irm_type'] == 'exponential':
#             loss = penalty_weight * penalty + torch.log(loss)
#             loss = torch.log( 1 + torch.exp(loss)) if loss < 15 else loss    
#         return loss 

#     def update(self, data_envs, unlabeled=None):

#         nll, penalty, logits_var = self.compute_loss(data_envs)
#         loss = self.compute_loss_with_penalty(nll, penalty, logits_var)

#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             # Reset Adam, because it doesn't like the sharp jump in gradient magnitudes that happens at this step.
#             self.optimizer = get_optimizer(self.hparams, self.network.parameters())
            
#         self.optimizer.zero_grad()
#         loss.backward()
        
#         grad_norm = get_grad_norm(self.network.parameters())
#         clip_grad_norm_(self.network.parameters(), self.grad_clip)
#         self.optimizer.step()
#         # if self.hparams['set_imag_to_zero']:
#         #     set_imag_to_zero(self.network.parameters())
        
#         w_norm = get_weight_norm(self.network.parameters())
#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),  'penalty': penalty.sqrt().item(), 
#                 'grad_norm': grad_norm.item(), 'weight_norm': w_norm.item(), 'logits_amp': logits_var.item(),}    


class IRM(ERM):

    const_type = 'IRM'
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
        self.optimizer = get_optimizer(self.hparams, self.network.parameters())
        # self.optimizer_ERM = get_optimizer(self.hparams, self.get_params_list('others'))
        # self.optimizer_penalty = get_optimizer(self.hparams, self.get_params_list('multiplicative'))
        self.update_count = 0

    # def get_params_list(self, type):
    #     if type == 'multiplicative':
    #         return [param for param in self.network.featurizer.parameters() if len(param.shape) == 0]
    #     else: 
    #         return [param for param in self.network.featurizer.parameters() if len(param.shape) != 0] + list(self.network.classifier.parameters())
    
    def loss_fn_IRM(self, logits, label_all):
        if len(label_all.shape) == 2:
            label_all=label_all.unsqueeze(-1)
        if len(logits.shape ) == 2:
            logits=logits.unsqueeze(-1)
            
        # n_env, batch, n_out = label_all.shape;        assert n_out == 1
        L0 = self.get_L0(logits, label_all)
        constraint = self.get_constraint(logits, label_all)
        penalty = constraint.abs().pow(2).mean()
        
        # penalty_coeff = self.hparams['irm_lambda'] 
        penalty_coeff = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        
        if self.hparams['loss_type'] in ['regression', 'regression_complex']:
            penalty_coeff *= 4
        
        if self.hparams['irm_type'] in [None, 'additive']:
            loss = L0 + penalty_coeff * penalty
        elif self.hparams['irm_type'] == 'multiplicative':
            loss = L0 * (1 + penalty_coeff * penalty)
        elif self.hparams['irm_type'] == 'exponential':
            loss = penalty_coeff * penalty + torch.log(L0)
            loss = torch.log( 1 + torch.exp(loss)) if loss < 15 else loss
        
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
               
        const_type=self.const_type
        if const_type=='IRM':
            g = oo - oy
            return g
        else:
            if const_type == 'oy':
                g = oy
            elif const_type == 'oo':
                g = oo
            elif const_type == 'oo-oy':
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

        loss, constraint, L0 = self.loss_fn_IRM(logits, label_all) 
        logits_var = logits.var(keepdim=True, dim=(1,2)).mean().detach()
        
        # self.optimizer_ERM.zero_grad()
        # L0.backward()
        # grad_norm = get_grad_norm(self.get_params_list('others'))
        # clip_grad_norm_(self.get_params_list('others'), self.hparams['grad_clip'])
        # self.optimizer_ERM.step()
        # logits = self.predict(input_all);        logits = logits.view(label_all.shape)
        # loss, constraint, L0 = self.loss_fn_IRM(logits, label_all)
        # self.optimizer_penalty.zero_grad()
        # loss.backward()
        # grad_norm = get_grad_norm(self.get_params_list('multiplicative'))
        # clip_grad_norm_(self.get_params_list('multiplicative'), self.hparams['grad_clip'])
        # self.optimizer_penalty.step()
        
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
    const_type = 'oy'

# class ICM_oo(IRM):
#     const_type = 'oo'

class relaxed_IRM(IRM):
    const_type = 'oo-oy'
        

##########################################################

class IRM_ADMM(ERM):

    const_type = 'IRM'
    
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams, **kwargs)
        if self.const_type == 'IRM':
            num_constraint = num_domains
        else:
            num_constraint = num_domains-1
            self.U_diff = get_ortho_diff(num_domains)
            
        self.network.const_sum = getattr(self.network, 'const_sum', torch.zeros(num_constraint,1))
        # self.network.const_momentum = getattr(self.network, 'const_momentum', torch.zeros(num_constraint,1))
        self.optimizer = get_optimizer(self.hparams, self.network.parameters())
        self.update_count = 0
        self.network.scale = getattr(self.network, 'scale', torch.ones(1,requires_grad=True))

    def loss_fn_ADMM(self, logits, label_all): 
        self.network.const_sum = self.network.const_sum.to(logits.device, dtype=logits.dtype)
        # self.network.const_momentum = self.network.const_momentum.to(logits.device, dtype=logits.dtype)

        if len(label_all.shape) == 2:
            label_all=label_all.unsqueeze(-1)
        if len(logits.shape ) == 2:
            logits=logits.unsqueeze(-1)
            
        # n_env, batch, n_out = label_all.shape;        assert n_out ==1
        
        L0 = self.get_L0(logits, label_all)
        constraint = self.get_constraint(logits, label_all)
        penalty = (constraint + self.network.const_sum).abs().pow(2).mean() - self.network.const_sum.abs().pow(2).mean()
        
        penalty_coeff = self.hparams['ADMM_mu'] #  if 'anneal_iters' not in self.hparams or self.update_count >= self.hparams['anneal_iters'] else 0
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
               
        const_type=self.const_type
        if const_type=='IRM':
            g = oo - oy
            return g
        else:
            if const_type == 'oy':
                g = oy
            elif const_type == 'oo':
                g = oo
            elif const_type == 'oo-oy':
                g = oo - oy
            elif const_type == 'yy-oy':
                if self.hparams['loss_type'] in ['regression', 'regression_complex']:
                    g = yy - oy
                else:
                    g = -oy
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
            
        loss, constraint, L0 = self.loss_fn_ADMM(logits, label_all)
        
        # if self.hparams['grad_penalty']:
        #     loss = add_grad_penalty(loss, self.network.parameters(), self.hparams['grad_penalty'])
        
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
        
        # self.network.const_momentum = (1-self.hparams['ADMM_momentum']) * constraint  + self.hparams['ADMM_momentum']*self.network.const_momentum
        # self.network.const_sum += self.hparams['ADMM_accum_rate']*self.network.const_momentum
        
    def update_scale(self, logits, label_all):
        scale = self.network.scale.to(device=logits.device, dtype=logits.dtype)
        loss_scale = self.loss_fn(scale*logits.detach(), label_all)
        scale_grad = autograd.grad(loss_scale, self.network.scale, create_graph=False)
        self.network.scale = self.network.scale - self.hparams['lr_scale_grad']*scale_grad[0]; self.network.scale.grad=None
        return scale


class MRI_ADMM(IRM_ADMM):
    const_type = 'oy'

# class ICM_oo_ADMM(IRM_ADMM):
#     const_type = 'oo'

class relaxed_IRM_ADMM(IRM_ADMM):
    const_type = 'oo-oy'

# class rev_IRM_ADMM(IRM_ADMM):
#     const_type = 'yy-oy'
    
def get_ortho_diff(n_env):
    # import math
    ones = torch.ones(n_env,1) 
    a,_ = torch.linalg.qr(ones, mode='complete')
    # U_mean = a[0:1]  # ones.T /math.sqrt(n_env)
    U_diff = a[1:]
    return U_diff        