from typing import Any, Optional

import torch
import torch.nn as nn
import optuna

from train.trainer_rgbn_triplet import Trainer_RGBN_Triplet
from train.validation_only import Trainer_Validation_Only


def get_trainer(cfgs: dict,
                fabric: Any,
                model: nn.Module,
                train_loader: torch.utils.data.DataLoader = None,
                val_loader: torch.utils.data.DataLoader = None,
                optimizer: torch.optim.Optimizer = None,
                criterion: nn.Module = None,
                unique_dir_name: str = None,
                trial: Optional[optuna.trial.Trial] = None) -> Any:
    """ Gets the trainer class for the current configuration

    Args:
        cfgs: configuration dictionary
        fabric: fabric object
        model: model to train
        train_loader: training data loader
        val_loader: validation data loader
        optimizer: optimizer
        criterion: loss function

    Returns:
        Trainer class

    Raises:
        NotImplementedError: if trainer is not found

    """

    if cfgs.trainer_name == "trainer_rgbn_triplet":  # up to date
        trainer = Trainer_RGBN_Triplet(cfgs, fabric, model, train_loader,
                                       val_loader, optimizer, criterion,
                                       unique_dir_name, trial)
    elif cfgs.trainer_name == "validation_only":
        trainer = Trainer_Validation_Only(cfgs, fabric, model, val_loader)
    else:
        trainer = None

    # check if trainer was found
    if trainer is None:
        raise NotImplementedError("Trainer {} not found".format(
            cfgs.trainer_name))

    return trainer
