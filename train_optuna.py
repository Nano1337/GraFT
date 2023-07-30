import os
import time
import random
from pathlib import Path

import wandb
import optuna
from optuna.trial import Trial

import torch.utils.data
from lightning.fabric import Fabric, seed_everything

import models
import data_utils
import train 
import optimizers
import loss
import os
from utils.opt import get_cfgs


torch.set_float32_matmul_precision("medium")
# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

def objective(trial: Trial, cfgs: dict, fabric: Fabric, rand_seed: int):

    # suggest values of hyperparam using a trial object
    hyperparams = {
        # "lr": trial.suggest_float("lr", 8e-6, 1e-5, log=False),
        "lr": trial.suggest_float("lr", 1e-5, 1e-5, log=False), # default
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.001, log=False),
        "beta1": trial.suggest_float("beta1", 0.80, 0.85),
        "beta2": trial.suggest_float("beta2", 0.97, 0.98)
        # "warmup_steps": trial.suggest_int("warmup_steps", 100, 800),
        # "cycles": trial.suggest_float("cycles", 0.1, 1.0),
    }

    # update cfgs with suggested hyperparameters
    for hyperparam, value in hyperparams.items():
        setattr(cfgs, hyperparam, value)

    if cfgs.wandb_trial_name:
        trial_name = cfgs.wandb_trial_name + "_trial_" + str(trial.number)
    else:
        trial_name = "trial_" + str(trial.number)

    if fabric.is_global_zero: 
        wandb.init(
            project=cfgs.wandb_project, 
            config=cfgs,
            group=cfgs.study_name,
            name=trial_name,
            reinit=True,
        )
        wandb.run.save()

        # log optuna hyperparams to wandb once
        for hyperparam, value in hyperparams.items():
            wandb.log({"hyperparams/"+str(hyperparam): value})
        wandb.log({"hyperparams/seed": rand_seed})

    # log trial number to cfgs
    cfgs.trial_number = trial.number

    highest_val = main(cfgs, fabric, trial)

    return highest_val

def main(cfgs: dict, fabric, trial: Trial = None): 

    # create the dataset and dataloader 
    train_dataset, val_dataset, train_loader, val_loader = data_utils.get_dataset_and_dataloader(cfgs)
    
    # Shape: batch_size, modalities, triplet, channels, height, width
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    dataset_size = len(train_dataset) + len(val_dataset)

    if fabric.is_global_zero: 
        print("Dataset size total:", dataset_size)
        print("Training set size:", len(train_dataset))
        print("Validation set size:", len(val_dataset))

    # create output directory
    config_name = os.path.splitext(os.path.basename(cfgs.cfg_name))[0]
    unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name

    output_dir = Path(cfgs.output_dir, unique_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir)
    

    # Get the model
    model = models.get_model(cfgs=cfgs, fabric=fabric)

    criterion = loss.get_loss(cfgs, fabric)

    optimizer = optimizers.get_optim(cfgs, model)

    trainer = train.get_trainer(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial)

    # print out model summary
    if fabric.is_global_zero: 
        trainer.print_networks()

    # begin training
    if cfgs.phase == "train":
        highest_val = trainer.train()
    elif cfgs.phase == "val":
        print(trainer.validate())

    return highest_val

if __name__ == "__main__":
    cfgs = get_cfgs()
    if cfgs.output_dir: 
        Path(cfgs.output_dir).mkdir(parents=True, exist_ok=True)

    # Set random seed for reproduceability
    if cfgs.new_random: 
        rand_seed = random.randint(0, 2**32 - 1)
    cfgs.seed = rand_seed
    seed_everything(cfgs.seed) 

    fabric = Fabric(accelerator="auto", devices = cfgs.gpus)
    fabric.launch()
    
    # set up optuna study
    sampler = optuna.samplers.QMCSampler(seed=cfgs.seed)
    # sampler = optuna.samplers.TPESampler(seed=cfgs.seed)
    study = optuna.create_study(
        direction="maximize", # maximize validation mAP
        study_name=cfgs.study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=1, 
            n_warmup_steps=100
            ),
        sampler=sampler,
        )  

    # run optuna
    study.optimize(
        lambda trial: objective(trial, cfgs, fabric, cfgs.seed), 
        n_trials=1, 
        n_jobs=1, 
        )

    # report results to console
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items(): 
        print("    {}: {}".format(key, value))

'''
For a complete random initial search, use:

Distributed training via spawning up multiple tmux sessions manually and calling the bash script
Multiple different random seeds are generated every run so there should be distributed uniform sampling of the hyperparameter space

'''