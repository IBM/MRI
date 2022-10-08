# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections
import json
import os
import time
import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from pytorch_lightning.loggers import TensorBoardLogger
    
from domainbed import datasets
import custom_datasets
datasets_dict = {**vars(datasets), **vars(custom_datasets)}

from custom_algorithms import get_algorithm_class 
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

from .arguments import get_args, get_args_str
from .utils import print_setting, print_args, seed_everything, set_directories, get_hparams
from .visualize_utils import plot_target_vs_output_angle


def get_Logger(hparams, tensorboard_dir):
    logger = TensorBoardLogger(tensorboard_dir, name='', version=None)
    logger.log_hyperparams(hparams) 
    logger.save() 
    summary_writer = logger.experiment
    return summary_writer

def get_algorithm(params, algorithm_dict=None, network=None):
    algorithm_class = get_algorithm_class(params['algorithm'])
    algorithm = algorithm_class(params['input_shape'], params['num_classes'], params['num_domains'], params['hparams'], network=network)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)
        
    algorithm.start_step = 0
    algorithm.config_num = 0
    algorithm.tensorboard_dir = params['tensorboard_dir']
    algorithm.device = params['device']   
    algorithm.params = params
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


def get_get_loaders(splits, test_envs, batch_size, eval_batch_size, num_workers, replacement, shuffle):
    def get_loaders(batch_size=batch_size, eval_batch_size=eval_batch_size, replacement=replacement, shuffle=shuffle):
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

    seed_everything(args.seed)
    assert args.dataset in datasets_dict

    hparams = get_hparams(args)
    tensorboard_dir = set_directories(args)    
    dataset = datasets_dict[args.dataset](args.data_dir, args.test_envs, hparams)  
    splits = split_dataset_(args, dataset, hparams['class_balanced'])
    get_loaders = get_get_loaders(splits, args.test_envs, batch_size=hparams['batch_size'], 
                                  eval_batch_size=args.eval_batch_size, num_workers=dataset.N_WORKERS,
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
    
    results = collections.defaultdict(list)
    
    def save_checkpoint(filename):
        if not args.skip_model_save:
            torch.save(algorithm.cpu().state_dict(), os.path.join(tensorboard_dir, filename))
    
    
    if args.fixed_linear_classifier_weights:
        if ((args.dataset.startswith('ShapeTexture') and args.feature_type == 'factors')
            or args.dataset == 'toyCMNIST'):
            assert params['hparams']['featurizer_type'] == 'identity', "featurizer_type should be identity"
            assert params['hparams']['classifier_type'] == 'linear', "classifier_type should be linear"
            
            param1, param2 = args.fixed_linear_classifier_weights
            for name, param_ in algorithm.network.named_parameters():
                if name.startswith('classifier'):
                    if args.loss_type == 'regression_complex':
                        param_.data = torch.tensor([[param1+0j, param2+0j]], dtype=torch.cfloat)
                    elif args.loss_type == 'binary_classification':
                        param_.data = torch.tensor([[param2, param1]], dtype=torch.float)
        
    ################### save config file
    algorithm.config_num += 1
    config_path = os.path.join(tensorboard_dir, f'config{algorithm.config_num}.json')

    config = {'params': params,  #'hparams': hparams,
              'args': vars(args)}
    with open(config_path, 'a') as f:
        f.write(json.dumps(config, sort_keys=True) + "\n")

    if args.load_model_path:
        print('loading model')
        algorithm.network.load_state_dict(torch.load(args.load_model_path))
        
    device = algorithm.device
    algorithm.to(device)
        
    train_loaders, uda_loaders, evals = loaders
    train_loaders_iter = zip(*train_loaders)
    uda_loaders_iter = zip(*uda_loaders) if uda_loaders is not None else None

    ################### Start Training
    start_step = algorithm.start_step
    
    for step in tqdm.tqdm(range(start_step, start_step + args.steps)):
        unlabeled_data = [x.to(device) for x, _ in next(uda_loaders_iter)] if args.task == "domain_adaptation" and uda_loaders_iter is not None else None
        train_data = [(x.to(device), y.to(device)) for x, y in  next(train_loaders_iter)]  # list of data batch over environments
        step_vals = algorithm.update(train_data, unlabeled=unlabeled_data)

        for key, val in step_vals.items():
            summary_writer.add_scalar(f'step_vals/{key}', np.mean(val), step)
            
        if (step % args.checkpoint_freq == 0) or (step == args.steps - 1):
            summary_writer.add_scalar("epoch", step / args.steps_per_epoch, step)
            
            if params['hparams']['featurizer_type'] == 'identity':
                weights = algorithm.network.classifier[-1].weight
                weights = weights.view(-1,weights.shape[0])
                for i in range(weights.shape[0]):
                    for j in range(weights.shape[1]):
                        summary_writer.add_scalar('weight/'f'[{i},{j}]', weights[i,j].item().real, step)
                        results['weight/'f'[{i},{j}]'].append(weights[i,j].item().real)
                        if weights.dtype==torch.cfloat:
                            summary_writer.add_scalar('weight/'f'[imag{i},{j}]',
                                                      weights[i,j].item().imag, step)
                            results['weight/'f'[imag{i},{j}]'].append(weights[i,j].item().imag)
                     
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
                    results[get_name('acc')].append(acc)
                    results[get_name('loss')].append(loss)
                    
                    if args.dataset.startswith("ShapeTexture") and args.loss_type in ['regression_complex']:
                        add_scatter_plots(summary_writer, logits, label, step, get_name)

                    # summary_writer.add_scalar('mem_gb', torch.cuda.max_memory_allocated() / (1024.*1024.*1024.), step)

                if hasattr(algorithm, 'get_constraint'):
                    # logits_all = torch.stack(logits_all); label_all = torch.stack(label_all);
                    min_batch_size = min(
                        [logits_all_.shape[0] for logits_all_ in logits_all]
                    )
                    logits_all = torch.stack(
                        [logits_all_[:min_batch_size] for logits_all_ in logits_all]
                    )
                    label_all = torch.stack(
                        [label_all_[:min_batch_size] for label_all_ in label_all]
                    )
                    constraint = algorithm.get_constraint(logits_all, label_all)
                    summary_writer.add_scalar(f'{key}/penalty', constraint.norm().item(), step)
                
            algorithm.start_step = step + 1

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
                
    summary_writer.close()
    save_checkpoint(f'model{algorithm.config_num}.pkl')
    return results


def add_scatter_plots(summary_writer, logits, label, step, get_name, num_scatter_points = 500):
    fig1, fig2 = plot_target_vs_output_angle(label[:num_scatter_points],  logits[:num_scatter_points], step)
    summary_writer.add_figure(get_name('real_imag_plots'), fig1, step)
    summary_writer.add_figure(get_name('target_output_plots'), fig2, step)

def main(args_str=None, hparams_str=None, algorithm=None, network=None):
    args = get_args() if args_str is None else get_args_str(args_str, hparams_str)
    args, params, get_loaders = initialize(args)
    loaders = get_loaders()
    algorithm = algorithm or get_algorithm(params, network=network)
    results = train(args, params, algorithm, loaders)
    return args, params, loaders, algorithm, results


if __name__ == "__main__":
    main()