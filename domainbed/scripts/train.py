# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import json
import os
import time
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
# import pandas as pd
# import matplotlib.pyplot as plt
# import imageio
# import math
    
from domainbed import datasets
import custom_datasets
datasets_dict = {**vars(datasets), **vars(custom_datasets)}   #datasets_dict = vars(datasets)

from custom_algorithms import get_algorithm_class    # from domainbed.algorithms import get_algorithm_class
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
# from datamodules.custom_loader import InfiniteDataLoader_Huh

from .arguments import get_args, get_args_str
from .utils import print_setting, print_args, seed_everything, set_directories, get_hparams

from visualize_utils import plot_target_vs_out_angle2

# from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.loggers import TensorBoardLogger

def get_Logger(hparams, tensorboard_dir):
    # os.makedirs(tensorboard_dir, exist_ok=True)
    # summary_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger = TensorBoardLogger(tensorboard_dir, name='', version=None) #hparams.v_num) #, default_hp_metric = False) 
    logger.log_hyperparams(hparams) # save hparams
    logger.save()     # save_hparams_to_yaml(os.path.join(hparams.save_dir,'hparams.yaml'), hparams)   

    summary_writer = logger.experiment
    return summary_writer

def get_algorithm(params, algorithm_dict=None, network=None): #, loss_fn=None):
    algorithm_class = get_algorithm_class(params['algorithm'])
    algorithm = algorithm_class(params['input_shape'], params['num_classes'], params['num_domains'], params['hparams'], network=network) #, loss_fn = loss_fn)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
        
    algorithm.start_step = 0
    algorithm.config_num = 0
    algorithm.tensorboard_dir = params['tensorboard_dir']
    algorithm.device = params['device']   
    algorithm.params = params

    if params['hparams'].get('initialize_weights_zero',False):
        nn.init.constant_(algorithm.network.classifier[-1].weight, 0.0)
    if params['hparams'].get('fixed_linear_weights',False):
        param1, param2 = params['hparams']['fixed_linear_weights']
        algorithm.network.classifier[-1].weight = nn.Parameter(torch.tensor([[param1+0j, param2+0j]], 
                                                                        device=algorithm.device,
                                                                        dtype=torch.cfloat))
    if params['hparams'].get('init_scale',1) != 1:
        def init_scale(module):
            if type(module) == nn.Linear:
                # print(module.weight.data)
                module.weight.data *= params['hparams']['init_scale']
                # print(module.weight.data)
                
        algorithm.network.apply(init_scale)

    return algorithm


def split_dataset_(args, dataset, balanced):
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.

    in_splits = []
    out_splits = []
    uda_splits = []

    for env_id, env in enumerate(dataset):

        out, in_ = misc.split_dataset(env,
                                      max(1, int(len(env) * args.holdout_fraction)),
                                      misc.seed_hash(args.trial_seed, env_id))
        if args.uda_holdout_fraction>0:
            if env_id in args.test_envs:
                uda, in_ = misc.split_dataset(in_,
                                            int(len(in_) * args.uda_holdout_fraction),
                                            misc.seed_hash(args.trial_seed, env_id))
            uda_weights = misc.make_weights_for_balanced_classes(uda) if balanced else None
            uda_splits.append((uda, uda_weights, env_id))  # Unlabeled test dataset (for adaptation)

        in_weights = misc.make_weights_for_balanced_classes(in_) if balanced else None
        in_splits.append((in_, in_weights, env_id))

        out_weights = misc.make_weights_for_balanced_classes(out) if balanced else None
        out_splits.append((out, out_weights, env_id))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    if len(uda_splits)==0:
        uda_splits = None 

    return {'in':in_splits, 'out':out_splits, 'uda':uda_splits}


def get_get_loaders(splits, test_envs, batch_size, num_workers, replacement, shuffle):
    def get_loaders(batch_size=batch_size, eval_batch_size=3000, replacement=replacement, shuffle=shuffle):
        train_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights, batch_size=batch_size, num_workers=num_workers, replacement=replacement, shuffle=shuffle)
                         for env, env_weights, env_id in splits['in'] if env_id not in test_envs]

        if splits['uda'] is not None:
            uda_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights, batch_size=batch_size, num_workers=num_workers, replacement=replacement, shuffle=shuffle)
                        for env, env_weights, env_id in splits['uda']] # if env_id in test_envs]
        else:
            uda_loaders = None

        evals = {key: [{'loader': FastDataLoader(dataset=env, batch_size=eval_batch_size, num_workers=num_workers),
                        'weights': None,
                        'env_id': env_id }  for env, env_weights, env_id in split]
                for key, split in splits.items() if split is not None}

        return (train_loaders, uda_loaders, evals)

    return get_loaders


def initialize(args): 
    # to implement checkpointing, just persist algorithm_dict
    # every once in a while, and then load them from disk here.

    seed_everything(args.seed)
    assert args.dataset in datasets_dict

    hparams = get_hparams(args)
    tensorboard_dir = set_directories(args)    
    dataset = datasets_dict[args.dataset](args.data_dir, args.test_envs, hparams)  
    splits = split_dataset_(args, dataset, hparams['class_balanced'])
    get_loaders = get_get_loaders(splits, args.test_envs, batch_size=hparams['batch_size'], num_workers=dataset.N_WORKERS,
                                  replacement=args.replacement, shuffle=args.shuffle)

    if args.loss_type == 'binary_classification':
        if dataset.num_classes == 1:
            pass
        elif dataset.num_classes == 2:
            dataset.num_classes == 1
        else:
            raise ValueError('dataset.num_classes should be 1 for binary_classification')

    params = {  #"args": args, #vars(args),
                "input_shape": dataset.input_shape,
                "num_classes": dataset.num_classes,
                "num_domains": len(dataset) - len(args.test_envs),
                "hparams": hparams, 
                "algorithm": args.algorithm, 
                "device": "cuda:" + args.which_gpu if torch.cuda.is_available() else "cpu",
                "tensorboard_dir": tensorboard_dir}
        
    args.steps_per_epoch = min([len(env) // hparams['batch_size'] for env, _, _ in splits['in']])
    if args.steps_per_epoch==0:
        raise ValueError()
        
    args.steps = args.steps or dataset.N_STEPS
    args.checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    
    if args.print_args:
        print_setting()
        print_args(args, hparams)

    return args, params, get_loaders


def train(args, params, algorithm, loaders):
    tensorboard_dir = algorithm.tensorboard_dir
    summary_writer = get_Logger(params['hparams'], tensorboard_dir)

    # if args.save_results:
    # results = collections.defaultdict(list)
    
    def save_checkpoint(filename):
        if not args.skip_model_save:
            torch.save(algorithm.cpu().state_dict(), os.path.join(tensorboard_dir, filename))

    ################### save config file
    algorithm.config_num += 1
    config_path = os.path.join(tensorboard_dir, f'config{algorithm.config_num}.json')

    config = {'params': params,  #'hparams': hparams,
              'args': vars(args)}
    with open(config_path, 'a') as f:
        f.write(json.dumps(config, sort_keys=True) + "\n")


    device = algorithm.device
    algorithm.to(device)
        

    train_loaders, uda_loaders, evals = loaders
    train_loaders_iter = zip(*train_loaders)
    uda_loaders_iter = zip(*uda_loaders) if uda_loaders is not None else None

    ################### Start Training
    start_step = algorithm.start_step
    # last_results_keys = None
    # checkpoint_vals = collections.defaultdict(list)
    
    for step in tqdm.tqdm(range(start_step, start_step + args.steps)):
        # step_start_time = time.time()
        unlabeled_data = [x.to(device) for x, _ in next(uda_loaders_iter)] if args.task == "domain_adaptation" and uda_loaders_iter is not None else None
        train_data = [(x.to(device), y.to(device)) for x, y in  next(train_loaders_iter)]  # list of data batch over environments
        step_vals = algorithm.update(train_data, unlabeled=unlabeled_data)
        #         checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            # checkpoint_vals[key].append(val)
            summary_writer.add_scalar(f'step_vals/{key}', np.mean(val), step)
        
        if params['hparams']['featurizer_type'] == 'identity':
            weights = algorithm.network.classifier[-1].weight
            weights = weights.view(-1,weights.shape[0])
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    summary_writer.add_scalar('weight/'f'[{i},{j}]', weights[i,j].item().real, step)
                    # if (step % args.checkpoint_freq == 0) or (step == args.steps - 1):
                    #     results['weight/'f'[{i},{j}]'].append(weights[i,j].item().real)
                    if weights.dtype==torch.cfloat:
                        summary_writer.add_scalar('weight/'f'[imag{i},{j}]', weights[i,j].item().imag, step)
                        # if (step % args.checkpoint_freq == 0) or (step == args.steps - 1):
                        #     results['weight/'f'[imag{i},{j}]'].append(weights[i,j].item().imag)
            
        if (step % args.checkpoint_freq == 0) or (step == args.steps - 1):
            summary_writer.add_scalar("epoch", step / args.steps_per_epoch, step)
            
            # results['step_number'].append(step)
            # import pdb; pdb.set_trace()
            penalty_ = {}
            
            for key, eval_list in evals.items():
                logits_all, label_all = [], []
                for eval_dict in eval_list:
                    env_id, loader, eval_weights = eval_dict['env_id'], eval_dict['loader'], eval_dict['weights']
                    def get_name(str):
                        return f'{key}/{str}/env{env_id}'

                    acc, loss, label, logits = misc.accuracy(algorithm,loader,eval_weights,device,args.loss_type)
                    
                    if env_id not in args.test_envs:
                        logits_all.append(logits); label_all.append(label)
                    
                    summary_writer.add_scalar(get_name('acc'), acc, step)
                    summary_writer.add_scalar(get_name('loss'), loss, step)
                    # results[get_name('acc')].append(acc)
                    # results[get_name('loss')].append(loss)
                    
                    if args.dataset.startswith("ShapeTexture") and args.loss_type in ['regression_complex']:
                        add_scatter_plots(summary_writer, logits, label, step, get_name)

                    # if (step == 0) or (step == args.steps - 1):
                        # summary_writer.add_scalar('mem_gb', torch.cuda.max_memory_allocated() / (1024.*1024.*1024.), step)

                if hasattr(algorithm, 'get_constraint'):
                    logits_all = torch.stack(logits_all); label_all = torch.stack(label_all);
                    constraint = algorithm.get_constraint(logits_all, label_all)
                    penalty_[key] = constraint.norm().item()
                    summary_writer.add_scalar(f'{key}/penalty', penalty_[key], step)
                    
            if hasattr(algorithm, 'get_constraint'):
                summary_writer.add_scalar(f'log_penalty_ratio', np.log(penalty_['out']/penalty_['in']), step)
                
            algorithm.start_step = step + 1
            # checkpoint_vals = collections.defaultdict(list)

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
                
    summary_writer.close()
    save_checkpoint(f'model{algorithm.config_num}.pkl')
    # if args.save_results:
    #     saved_results = pd.DataFrame(results)
    #     saved_results.to_csv(tensorboard_dir+'/results.csv', index = False)
    # return results

#     with open(os.path.join(tensorboard_dir, 'done'), 'w') as f:
#         f.write(f'done{algorithm.config_num}')


def add_scatter_plots(summary_writer, logits, label, step, get_name, num_scatter_points = 500):
    
    fig1, fig2 = plot_target_vs_out_angle2(label[:num_scatter_points],  logits[:num_scatter_points], step)
    summary_writer.add_figure(get_name('real_imag_plots'), fig1, step)
    summary_writer.add_figure(get_name('target_output_plots'), fig2, step)
    
    # if (step == 0) or (step == args.steps - 1):
    #     fig1.savefig(tensorboard_dir+f'/real_imag_{step}.pdf', bbox_inches='tight')
    #     fig2.savefig(tensorboard_dir+f'/target_output_{step}.pdf', bbox_inches='tight')


def main(args_str=None, hparams_str=None, algorithm=None, network=None):
    
    args = get_args() if args_str is None else get_args_str(args_str, hparams_str)
    
    args, params, get_loaders = initialize(args)
    loaders = get_loaders()           # optional: (batch_size, eval_batch_size)
    algorithm = algorithm or get_algorithm(params, network=network) #, loaders=loaders) # optional: (algorithm_dict, network)
    # results = train(args, params, algorithm, loaders)
    train(args, params, algorithm, loaders)
    return args, params, loaders, algorithm #, results



###################################################
# load_model
# def load_model(model, ckpt_dir, v_num):

#     # if v_num is None:
#     #     return model, None, 0, None  # model, optimizer_state_dict, epoch0, loss_min
#     # else:
#         ckpt_file = get_latest_ckpt(ckpt_dir)
#         assert ckpt_file is not None
#         ckpt = torch.load(ckpt_file)
#         model.load_state_dict(ckpt['model_state_dict'])
#         return model, ckpt['optimizer_state_dict'], ckpt['epoch'], ckpt['test_loss']

# ###############################################

# import glob
# import os

# def get_ckpt_dir(log_dir):
#     ckpt_dir = os.path.join(log_dir, 'checkpoints') 
#     if not os.path.isdir(ckpt_dir):
#         os.makedirs(ckpt_dir)            
#     return ckpt_dir

# def get_latest_ckpt(ckpt_path):
#     # ckpt_path = os.path.join(save_dir, 'checkpoints', '*')
#     filename_list = glob.glob(os.path.join(ckpt_path, '*'))
#     name_sorted = sorted(
#         filename_list, 
#         # key = lambda f: (int(f.split('-')[0].split('=')[1])), #, int(f.split('-')[1].split('=')[1].split('.')[0])),
#         key = lambda f: int(f.split('-')[0].split('=')[1].split('.')[0]),
#         reverse = True
#     )
#     return name_sorted[0] if len(name_sorted)>0 else None

# #############################

# def save_cktp(ckpt_dir, filename, epoch, test_loss, train_loss = None, model_state_dict = None, optimizer_state_dict = None):
#     torch.save({
#                 'epoch': epoch,
#                 'train_loss': train_loss,
#                 'test_loss': test_loss,
#                 'model_state_dict': model_state_dict,
#                 'optimizer_state_dict': optimizer_state_dict, #optimizer.state_dict() if optimizer is not None else None,
#                 }, os.path.join(ckpt_dir,filename))

# import shutil

# def _del_file(filepath):
#     if filepath is None:
#         pass
#     else:
#         dirpath = os.path.dirname(filepath)
#         # make paths
#         os.makedirs(dirpath, exist_ok=True)
#         try:
#             shutil.rmtree(filepath)
#         except OSError:
#             os.remove(filepath)



if __name__ == "__main__":
    main()