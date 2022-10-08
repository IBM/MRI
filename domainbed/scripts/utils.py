import os, sys
import random
import numpy as np
import torch
import torchvision
import PIL


def print_setting():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

def print_args(args, hparams):
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

def seed_everything(seed):
    if seed>0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
import re

def set_directories(args):
    dir_path = 'domainbed'     # dir_path = os.path.dirname(os.path.realpath(__file__))

    # os.makedirs(args.output_dir, exist_ok=True)
#     sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
#     sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # initializing tensorboard directory
    hparams_dir = None
    if args.hparams:
        hparams_dir = re.sub(r"('|{|}| )", "", args.hparams)
        hparams_dir = re.sub(r'"', '', hparams_dir)
        hparams_dir = re.sub(r",", "|", hparams_dir)
    temp_dir = os.path.join(dir_path,
                            args.output_dir, 
                            args.exp, #'tensorboard_summary',
                            args.dataset, args.algorithm, 
                            hparams_dir or 'default')
    # current_run = 0
    # while True:
    #     tensorboard_dir = os.path.join(temp_dir, f'run{current_run}')
    #     current_run += 1
    #     if not os.path.exists(tensorboard_dir):
    #         break
    # return tensorboard_dir
    return temp_dir


from domainbed import hparams_registry
import json

def get_hparams(args):
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
        
    if args.hparams is None:
        hparams_ = {}
    else:
        if isinstance(args.hparams,dict):
            hparams_ = args.hparams
        elif isinstance(args.hparams,str):
            hparams_ = json.loads(args.hparams)
        else:
            raise ValueError()
            
    # hparams_.update({key: getattr(args, key) for key in ['feature_type', 'loss_type', 'optim', 'grad_clip', 'momentum']})
    hparams_.update({key: getattr(args, key) for key in ['loss_type', 'optim',]})
    
    if args.dataset.startswith('ShapeTexture') or args.dataset.endswith('MNIST'):
        hparams_.update({key: getattr(args, key) for key in ['env_param_list', 
                                                             'causal_param', 
                                                             'total_batch']})
        
    if args.dataset.startswith('ShapeTexture'):
        hparams_.update({key: getattr(args, key) for key in ['max_phase',
                                                             'label_type',
                                                             'feature_type',
                                                             'n_bin']})
        
    hparams.update(hparams_)
    
    hparams_dict = {key: val for key,val in hparams_.items() if key not in ['grad_clip', 'momentum',
                                                                            'max_phase',
                                                                            'label_type',
                                                                            'n_bin',
                                                                           ] and not isinstance(val,str)}
    hparams_dict = {key: hparams_[key] for key in sorted(hparams_dict.keys(), key=lambda x:x.lower())}
    hparams_str = [val for key,val in hparams_.items() if isinstance(val,str)]
    args.hparams = str(hparams_str) + ',' + str(hparams_dict)
    return hparams

