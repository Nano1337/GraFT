#!/bin/bash

# Generate a random seed from the current nanoseconds
SEED=$(date +%N)

# Run the Python script with the --seed flag set to the random number
python train.py -c cfgs/deit_gradual_fusion_rnt100.yaml #--seed $SEED
