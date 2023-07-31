from typing import Any, Dict, Type
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Module
from abc import ABC, abstractmethod


class Base_Trainer(Module, ABC):
    """
    Base class for trainers.

    Args:
        cfgs (Dict): A dictionary containing the configuration parameters.
        fabric (Any): The computational device to use for calculations.
        model (Type[Module]): The PyTorch model to train.
        train_loader (Type[DataLoader]): The DataLoader for the training data.
        val_loader (Type[DataLoader]): The DataLoader for the validation data.
        optimizer (Type[optim.Optimizer]): The optimizer for training the model
        criterion (Type[Module]): The loss function used for training.
    """

    def __init__(self,
                 cfgs: Dict,
                 fabric: Any,
                 model: Type[Module],
                 train_loader: Type[DataLoader] = None,
                 val_loader: Type[DataLoader] = None,
                 optimizer: Type[optim.Optimizer] = None,
                 criterion: Type[Module] = None):
        super().__init__()
        self.cfgs = cfgs
        self.fabric = fabric
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

        self.lr_scheduler = None
        self.loaded_epoch = 0
        self.epoch = 0
        self.highest_val_acc = 0.0

    @abstractmethod
    def train_epoch(self, epoch: int, end_epoch: int, total_iters: int,
                    epoch_iter: int, iter_data_time: float,
                    optimize_time: float):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def save_networks(self, save_dir: str, save_name: str):
        pass

    @abstractmethod
    def load_networks(self, load_dir: str, load_name: str):
        pass

    def print_networks(self):
        print('---------- Networks initialized -------------')
        total_params = 0
        for name, module in self.model.named_children():
            for n, p in module.named_parameters():
                if p.requires_grad:
                    num_params = p.numel()
                    total_params += num_params

        if self.cfgs.verbose:
            print(self.model)

        if len(self.cfgs.gpus) > 1:
            print(
                f'Total number of parameters: {(total_params/2) / 1e6:.3f} M')
        else:
            print(f'Total number of parameters: {total_params / 1e6:.3f} M')
        print('-----------------------------------------------')
