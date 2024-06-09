# Benchmark of knowledge-augmented pre-trained language models for biomedical relation extraction

This repository contains source code to run benchmarks for knowledge-augmented pre-trained language models 
for biomedical relation extraction.

----
## Installation
First, download the repository and change into the directory.
```
git clone https://github.com/mariosaenger/biore-kplm-benchmark
cd biore-kplm-benchmark
```

Setup a virtual environment, using conda (or a framework of your choice):
```
conda create -n biore-kplm
conda activate biore-kplm
```
Install all necessary packages:
```
pip install -r requirements.txt
```

----
## Usage
### Experiment configuration

The code uses [Hydra](https://hydra.cc/) for experiment configuration and grid search for hyperparameter 
evaluation. The default configuration is given in `_configs/config.yaml`. Each subfolder in `_configs` 
contains alternative configurations for different experimental aspects:
- `callbacks`: Callbacks (e.g. checkpointing) to be used during experiment execution
- `context_info`: Configurations of context information to be used
- `data`: Dataset for which the benchmark should be executed
- `hydra`: Configuration options of the Hydra framework (e.g. output and logging directory) 
- `logger`: Logger (e.g., csv, wandb, comet) to used during experiment execution
- `model`: Model to be tested
- `trainer`: Options for the trainer (e.g., cpu or gpu) used

All configurations can also be overridden while calling the program (see [Hydra reference manual](https://hydra.cc/docs/intro/))

### Experiment execution

Experiments can be executed (using the configuration in `_configs/config.yaml`) with:
```
python -m kplmb.train
```
Default configuration options can be overridden via program parameters:
```
python -m kplmb.train model=pubmedbert-ft model.lr=3e-5 batch_size=16
```
To run multiple experiments at once `--multi-run` can be used. For instance, the following call 
runs 18 experiments testing 2 different learning rates, 3 different batch sizes and 3 different max lengths:
```
python -m kplmb.train --multirun \
	model=pubmedbert-ft \
	model.lr=3e-5,5e-5 \
	model.max_length=256,384,512 \
	batch_size=8,16,32
```

For the available configuration options see the configuration files in `_configs`.
