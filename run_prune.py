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

def count_zeros(parameters_to_prune):
    zero = total = 0
    for module, _ in parameters_to_prune:
        zero += float(torch.sum(module.weight == 0))
        total += float(module.weight.nelement())
    return zero, total

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
    for param in model.transformer.parameters():
        param.requires_grad = True
    model.transformer.pooler.dense.bias.requires_grad = False
    model.transformer.pooler.dense.weight.requires_grad = False

    parameters_to_prune = []
    for layer in model.transformer.encoder.layer:
        parameters_to_prune.append((layer.intermediate.dense, 'weight'))
        parameters_to_prune.append((layer.output.dense, 'weight'))
        parameters_to_prune.append((layer.attention.attention.query, 'weight'))
        parameters_to_prune.append((layer.attention.attention.key, 'weight'))
        parameters_to_prune.append((layer.attention.attention.value, 'weight'))
        parameters_to_prune.append((layer.attention.output.dense, 'weight'))
    print('Number of Linear Layers in Model:', len(parameters_to_prune))

    # create output directory
    config_name = os.path.splitext(os.path.basename(cfgs.cfg_name))[0]
    unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name

    output_dir = Path(cfgs.output_dir, unique_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir)

    if cfgs.use_wandb and fabric.is_global_zero:
        # WandB – Watch the model
        wandb.watch(model)

    criterion = loss.get_loss(cfgs, fabric)

    optimizer = optimizers.get_optim(cfgs, model)
    trainer = train.get_trainer(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name)
    #Pruning Hyperparams:
    amount = .7
    max_pruning_iterations = 30
    print('Entering Prune Algorithm')
    for iter in range(max_pruning_iterations):

        if fabric.is_global_zero:
            wandb.init(
                project=cfgs.wandb_project,
                config=cfgs,
                group=cfgs.study_name,
                reinit=True,
            )
            wandb.run.save()

        #manually save model
        #unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name
        #trainer.save_networks(os.path.join(cfgs.ckpt_dir, unique_dir_name), 'prune.pth', iter)  
        
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=amount) 
        zeros, total = count_zeros(parameters_to_prune)
        print('Pruning Iteration ', iter, ', # Zeros = ', zeros, ', Total Prunable Params = ', total)

        trainer.train()   
        
    

if __name__ == "__main__":
    cfgs = get_cfgs()
    if cfgs.output_dir: 
        Path(cfgs.output_dir).mkdir(parents=True, exist_ok=True)
    main(cfgs)

