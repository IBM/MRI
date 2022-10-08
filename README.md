# MRI
This repository contains source code for the paper [The Missing Invariance Principle found –
the Reciprocal Twin of Invariant Risk Minimization](https://arxiv.org/abs/2205.14546).

## Training

Refer [here](https://github.com/benhuh/MRI/blob/main/domainbed/scripts/train.py) for details on how to train a model. An example training procedure to train on the ColoredMNIST dataset is implemented in the python script below: 

```sh
from domainbed.scripts.train import main
exp = 'sample_training'
algorithm = 'MRI_ADMM'
dataset = 'CustomColoredMNIST'
loss_type = 'binary_classification'
optimizer = 'Adam'
momentum = 0.975
steps = 10000
batch_size = 256
ADMM_mu = 10
ADMM_accum_rate = 10
lr = 0.001
weight_decay = 0.001
featurizer = "LeNet"
args_str = f"--exp {exp} --algorithm {algorithm} --dataset {dataset} --loss_type {loss_type}\
             --optim {optimizer} --momentum {momentum} --steps {steps}"
hparams_str = {"batch_size":batch_size, "ADMM_mu":ADMM_mu, "ADMM_accum_rate":ADMM_accum_rate,
               "lr":lr, "weight_decay":weight_decay, "featurizer_type":featurizer}
main(args_str, hparams_str)
```

## Datasets
To run the binarized version of Terra Incognita and VLCS datasets used in this work, the user need to download it using this [link](https://zenodo.org/record/7146024).