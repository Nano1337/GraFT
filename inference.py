import os
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from lightning.fabric import Fabric, seed_everything

import models
import data_utils
import train 
import optimizers
import loss
from utils.opt import get_cfgs

torch.set_float32_matmul_precision("medium")


def main(cfgs: dict): 
    # Set random seed for reproduceability
    seed_everything(cfgs.seed)
    fabric = Fabric(accelerator="auto", devices=cfgs.gpus)
    fabric.launch()

    # create the dataset and dataloader 
    _, val_dataset, _, val_loader = data_utils.get_dataset_and_dataloader(cfgs)
    
    # Shape: batch_size, modalities, triplet, channels, height, width
    val_loader = fabric.setup_dataloaders(val_loader)

    dataset_size = len(val_dataset)

    if fabric.is_global_zero: 
        print("Validation set size:", dataset_size)
    
    # Get the model
    model = models.get_model(cfgs=cfgs, fabric=fabric)

    if cfgs.unfreeze:
        for param in model.transformer.parameters():
            param.requires_grad = True
        model.transformer.pooler.dense.bias.requires_grad = False
        model.transformer.pooler.dense.weight.requires_grad = False

    if cfgs.trainer_name != "validation_only":
        print("Warning: trainer_name is not validation_only. This script is only meant for validation.")

    trainer = train.get_trainer(cfgs=cfgs, fabric=fabric, model=model,
                                val_loader=val_loader)

    # print out model summary
    if fabric.is_global_zero:
        trainer.print_networks()

    trainer.validate()

if __name__ == "__main__":
    cfgs = get_cfgs()
    main(cfgs)
