# MRI
This anonymized repository contains source code for the paper 'The Missing Invariance Principle found –
the Reciprocal Twin of Invariant Risk Minimization'.

## Training

An example training procedure is implemented in the python script below: 

```sh
from domainbed.scripts.train import main

exp = 'sample_training'
algorithm = 'MRI_ADMM'
dataset = 'ColoredMNIST'
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
