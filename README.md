# Demo

This is the a anonymous repository of our article submitted in WWW 2025 "Model-heterogeneous Federated Learning with Spiking Neural
Network: An Empirical Study".


## Overview


## Requirements
```
torch==2.0.1
torchvision==0.15.2
spikingjelly==0.0.0.0.14
```

## How to run

run FL with SNNs under layer heterogeneity

```
python run_experiments.py -hete_config=./configs/stage_hete/vit4.yaml -n_parties=10 -frac=1 -local_epochs=3 -snn -log_round=1 -trs=True
```

run FL withs SNNs under ratio heterogeneity

```
python model_hete.py -trs=True -model=vit4 -strategy=hetero_fl -has_rate=True -config \ ./heterogeneity_configs/hetero_fl/vit4.yaml -snn -n_parties=10 -frac=1
```

We now release a simple demo of HeteroFL for FL with SNNs with SViT, more codes will be released after the acceptance of this paper.
