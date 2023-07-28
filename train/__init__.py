import importlib
import torch.nn as nn
from train.trainer_rgbn_triplet_verb import Trainer_RGBN_Triplet_Verb
from train.visualize_embedding_rnt import Trainer_Visualize_Embedding_RNT


def get_trainer(cfgs: dict, fabric, model: nn.Module, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial=None):
    """ Gets the trainer class for the current configuration 

    :param cfgs: configuration dictionary
    :param fabric: fabric object
    :param model: model to train
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param optimizer: optimizer
    :param criterion: loss function
    
    :return: Trainer class
    
    """

    if cfgs.trainer_name == "default": # up to date
        trainer = Trainer_1(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial)
    elif cfgs.trainer_name == "trainer_rgbn_triplet": # up to date
        trainer = Trainer_RGBN_Triplet_Verb(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial)
    elif cfgs.trainer_name == "visualize_embedding_rnt":
        trainer = Trainer_Visualize_Embedding_RNT(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial)
    else:
        trainer = None

    # check if trainer was found
    if trainer is None: 
        raise NotImplementedError("Trainer {} not found".format(cfgs.trainer_name))

    return trainer