import torch.nn as nn
from abc import ABC, abstractmethod

class Base_Trainer(nn.Module, ABC): 
    """
    Base class for trainers. This class is abstract and serves as the parent class for other trainer classes. It
    outlines the basic structure and functions a trainer should have.

    Args:
        cfgs (dict): A dictionary containing the configuration parameters.
        fabric (any): The computational device to use for calculations.
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        optimizer (Optimizer): The optimizer for training the model.
        criterion (nn.Module): The loss function used for training.

    Attributes:
        lr_scheduler (any, optional): Learning rate scheduler, if any.
        loaded_epoch (int): The epoch number from which training is to be resumed.
        epoch (int): The current epoch number during training.
        highest_val_acc (float): The highest validation accuracy achieved so far.
    """

    def __init__(self, cfgs, fabric, model, train_loader, val_loader, optimizer, criterion):
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
    def train_epoch(self, epoch, end_epoch, total_iters, epoch_iter, iter_data_time, optimize_time):
        """
        Abstract method to train the model for one epoch. Must be implemented by child classes.

        Args:
            epoch (int): The current epoch number.
            end_epoch (int): The final epoch number.
            total_iters (int): Total iterations to perform during training.
            epoch_iter (int): Number of iterations performed in the current epoch.
            iter_data_time (float): Time taken for data loading in the current iteration.
            optimize_time (float): Time taken for backpropagation and parameter update in the current iteration.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Abstract method to train the model. Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Abstract method to validate the model. Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def save_networks(self, save_dir, save_name):
        """
        Abstract method to save the model. Must be implemented by child classes.

        Args:
            save_dir (str): Directory to save the model.
            save_name (str): Name to save the model as.
        """
        pass

    @abstractmethod
    def load_networks(self, load_dir: str, load_name: str):
        """
        Abstract method to load the model. Must be implemented by child classes.

        Args:
            load_dir (str): Directory from which to load the model.
            load_name (str): Name of the model to load.
        """
        pass


    def print_networks(self):
        """
        Prints the total number of parameters in the model. If verbosity is turned on in the configurations,
        it also prints the architecture of the model.

        This method also checks if there's more than one GPU being used. If so, it prints the total parameters
        divided by two, else it prints the total parameters.
        """
        print('---------- Networks initialized -------------')
        total_params = 0
        for name, module in self.model.named_children():
            for n, p in module.named_parameters():
                if p.requires_grad:
                    num_params = p.numel()
                    # print(f"[Parameter {name}.{n}] Number of parameters: {num_params / 1e6:.3f} M")
                    total_params += num_params

        if self.cfgs.verbose:
            print(self.model)
        
        # check if there's more than one GPU being used
        if len(self.cfgs.gpus) > 1:
            print(f'Total number of parameters: {(total_params/2) / 1e6:.3f} M')
        else: 
            print(f'Total number of parameters: {total_params / 1e6:.3f} M')
        print('-----------------------------------------------')