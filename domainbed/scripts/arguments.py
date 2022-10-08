import argparse
import json

def get_args(*args):
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="./domainbed/data")
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,                      help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,      help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--which_gpu', type=str, default='0',       help='which gpu to use')
    parser.add_argument('--linear_featurizer', action='store_false')
    parser.add_argument('--identity_classifier', action='store_true')
    parser.add_argument('--feature_type', type=str, default='images',
                        help='feature type for shape texture dataset - images, factors or factored_nonlinear')
    parser.add_argument('--loss_type', type=str, default='classification', help='classification or regression')
    parser.add_argument('--optim', type=str, default='Adam',        help='Adam or SGD')
    parser.add_argument('--trial_seed', type=int, default=0,        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0,              help='Seed for everything else. Value 0 skips seeding. ')
    parser.add_argument('--steps', type=int, default=None,          help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--exp', type=str, default="temp")
    parser.add_argument('--no_replacement', action='store_false', dest='replacement')
    parser.add_argument('--shuffle_train', action='store_true', dest='shuffle')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0, help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--print_results', action='store_true', help="Print training results if true. Default is false.")
    parser.add_argument('--save_results', action='store_true',
        help="Save results.jsonl if true. Default is True.")
    parser.add_argument('--print_args', action='store_true',
        help="Do not print args and hparams if false. Default is true.") 
    parser.add_argument('--env_param_list', type=str, default=None)
    parser.add_argument('--causal_param', type=float, default=None)
    parser.add_argument('--total_batch', type=int, default=None)
    parser.add_argument('--max_phase', type=float, default=0)
    parser.add_argument('--n_bin', type=int, default=2)
    parser.add_argument('--label_type', type=str, default='shape')
    parser.add_argument('--fixed_linear_classifier_weights', type=str, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=3000)
    
    args_, unknown = parser.parse_known_args(*args) 
    
    if args_.env_param_list: args_.env_param_list = json.loads(args_.env_param_list)
    if args_.fixed_linear_classifier_weights: args_.fixed_linear_classifier_weights = json.loads(args_.fixed_linear_classifier_weights)
        
    return args_

def get_args_str(args_str, hparams_str):
    args = get_args(args_str.split())
    if hparams_str is not None:
        assert args.hparams is None
    args.hparams = hparams_str
    return args