#!/bin/bash

# Generate a random seed from the current nanoseconds
SEED=$(date +%N)

# Run the Python script with the --seed flag set to the random number
# python train_optuna.py -c cfgs/ablations/deit_vanilla_fusion_rnt100_stage1.yaml # --seed $SEED

export WANDB_API_KEY=b730655c05d6f0514256a0372293a335e8a5aa1a
export WANDB_DIR=/home/ubuntu/haoli3/research-GraFT/wandb

python train_optuna.py -c cfgs/ablations/deit_gradual_fusion_rnt100_stage2_random_step_ablation.yaml
