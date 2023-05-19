
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

To run the python compiler and compare with eager mode execution:

```bash
python torch2.py
```

To simply run the model for a number of steps and evaluate its execution time:

```bash
python RNN_AUV3D.py -r ../train_log/2023.04.24-11:25:38/
```
