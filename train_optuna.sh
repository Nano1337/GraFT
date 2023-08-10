#!/bin/bash

# Generate a random seed from the current nanoseconds
SEED=$(date +%N)

# Run the Python script with the --seed flag set to the random number
python train_optuna.py -c cfgs/deit_ortho_runs.yaml # --seed $SEED
