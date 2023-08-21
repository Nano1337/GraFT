#!/bin/bash

# Generate a random seed from the current nanoseconds
SEED=$(date +%N)

# Run the Python script with the --seed flag set to the random number
python train_optuna.py -c cfgs/deit_gradual_fusion_rnt100_stage2.yaml # --seed $SEED
# python train_optuna.py -c cfgs/deit_gradual_fusion_rn300_stage1.yaml