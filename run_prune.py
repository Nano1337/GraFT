import os
import time
import wandb
from pathlib import Path

import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from lightning.fabric import Fabric, seed_everything

from PIL import Image
from torchvision import transforms


import models
import data_utils
import train 
import optimizers
import loss
import numpy as np
from utils.opt import get_cfgs

#from prune_utils import unstructured_prune_model
import torch.nn.utils.prune as prune

torch.set_float32_matmul_precision("medium")

def main(cfgs: dict): 
    # Set random seed for reproduceability
    seed_everything(cfgs.seed)

    fabric = Fabric(accelerator="auto", devices = cfgs.gpus)
    fabric.launch()

    if cfgs.use_wandb and fabric.is_global_zero:
        # WandB – Initialize a new run
        wandb.init(project=cfgs.wandb_project, config=cfgs)
        wandb.run.name = cfgs.wandb_run_name
        wandb.run.save()

    # create the dataset and dataloader 
    train_dataset, val_dataset, train_loader, val_loader = data_utils.get_dataset_and_dataloader(cfgs)
    
    # Shape: batch_size, modalities, triplet, channels, height, width
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    dataset_size = len(train_dataset) + len(val_dataset)

    if fabric.is_global_zero: 
        print("Dataset size total:", dataset_size)
        print("Training set size:", len(train_dataset))
        print("Validation set size:", len(val_dataset))

    
    # Get the model
    model = models.get_model(cfgs=cfgs, fabric=fabric)

    if cfgs.use_wandb and fabric.is_global_zero:
        # WandB – Watch the model
        wandb.watch(model)

    criterion = loss.get_loss(cfgs, fabric)
    
    # print out model summary
    if fabric.is_global_zero: 
        trainer.print_networks()

    max_pruning_iterations = 10
    amount = .2

    parameters_to_prune = []
    for layer in model.transformer.encoder.layer:
        parameters_to_prune.append((layer.intermediate.dense, 'weight'))
        parameters_to_prune.append((layer.output.dense, 'weight'))
        parameters_to_prune.append((layer.attention.attention.query, 'weight'))
        parameters_to_prune.append((layer.attention.attention.key, 'weight'))
        parameters_to_prune.append((layer.attention.attention.value, 'weight'))
        parameters_to_prune.append((layer.attention.output, 'weight'))
    print('Number of Linear Layers in Model:', len(parameters_to_prune))


    for iter in range(max_pruning_iterations):
        # Make new output directory to save models
        config_name = os.path.splitext(os.path.basename(cfgs.cfg_name))[0]
        unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name

        output_dir = Path(cfgs.output_dir, unique_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Output directory:", output_dir)

        optimizer = optimizers.get_optim(cfgs, model)
        trainer = train.get_trainer(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name)

        #Init new wandb graph
        if cfgs.wandb_trial_name:
            trial_name = cfgs.wandb_trial_name + "_prune_iter_" + str(iter)
        else:
            trial_name = "prune_iter_" + str(iter)
            
        if fabric.is_global_zero:
            wandb.init(
                project=cfgs.wandb_project,
                config=cfgs,
                group=cfgs.study_name,
                name=trial_name,
                reinit=True,
            )
        wandb.run.save()

        print('Prune iter: ', iter)
        #prune_model(model, fabric, num_neurons_to_prune = num_neurons_to_prune)
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=amount)
        trainer.train()


    
    

if __name__ == "__main__":
    cfgs = get_cfgs()
    if cfgs.output_dir: 
        Path(cfgs.output_dir).mkdir(parents=True, exist_ok=True)
    main(cfgs)

