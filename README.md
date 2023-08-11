
# Requirments & Install:

## First create a conda environment with the right python version 

```bash
conda create -n {Name} python=3.9
```

The second step is to install [pypose](https://github.com/pypose/pypose) from source:

```bash
conda activate {Name}
```

clone the repo:
```bash
git clone  https://github.com/pypose/pypose.git
cd pypose
```

install the requirements
```bash
pip install -r requirements/runtime.txt
```

```bash
python setup.py develop
```

Test:
```bash
pytest
```

## Install the requriements for the module:

```bash
git clone https://github.com/NicolayP/rnn_auv.git
```

checkout to the desired branch:

```bash
git checkout optimization
```

install the requirements:

```bash
pip install -r requrements.txt
```


# usage:

To clean a set of bags before learning:

```bash
python clean_bag.py -d {bag_dir} -o {out_dir} -f {sample_frequency} -s {number of steps} -n {norm}
```


To run the python compiler and compare with eager mode execution:

```bash
python torch2.py
```


To simply run the model for a number of steps and evaluate its execution time:

```bash
python RNN_AUV3D.py -r ../train_log/2023.04.24-11:25:38/
```


# Structure

train-param: Contains yaml file defining the network topology, the min and max input. A list of the accepted entries are given in section 'training'

Data: contains the ros bags with the trajectories.

Scripts: Contain all the different python scripts


# Training:

```yaml
---
log:
    path: "." # logging directory.
    stamped: True # Wether or not to timestamp the log dir.

data_loader_params:
    batch_size: 512
    shuffle: True
    num_workers: 8

dataset_params:
    samples: 10  # the number of files to use for the learning. If not set, all the files in the dir are used.
    steps: 15 # The number of steps to use for the predictions
    dir: "." # Directory containing the csv files.
    split: 0.7 # Train/Validation split, 0.7 = 70% training, 30 validation.
    v_frame: "body" # Frame in which the velocity is expressed.
    dv_frame: "body" # Frame in which the velocity delta is expressed.
    act_normed: True # Normlaize action between [-1, 1].
    out_normed: False # Normlaize the targets between [-1, 1].

model:
    se3: True # If true, the network used the pypose library.
    rnn: # definition of the RNN part.
        rnn_layer: 1
        rnn_hidden_size: 64
        bias: True
        activation: "tanh"
    fc: # definition of the fully connected network.
        topology: [128, 128]
        activation: "LeakyRelu"
        bias: True
        batch_norm: True
        relu_neg_slope: 0.1

loss: # Loss function definition.
    type: "traj"
    traj: 1.
    vel: 1.
    dv: 1.

optim: # Optimizer definition.
    type: "Adam"
    lr: 1.e-4
    epochs: 4

actions_lim: # Defines limits on each action input.
    0: [-100, 100]
    1: [-100, 100]
    2: [-100, 100]
    3: [-100, 100]
    4: [-100, 100]
    5: [-100, 100]

```
