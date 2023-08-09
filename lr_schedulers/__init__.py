import torch.optim.lr_scheduler as lr_scheduler
from lr_schedulers.warmup_cosine_schedule import WarmupCosineSchedule
from lr_schedulers.warmup_sqrt_schedule import WarmupSqrtDecayWithPercentageCoolDown


def get_lr_scheduler(cfgs: dict, optimizer, train_loader):
    """
    Function to get the learning rate scheduler as specified by the configuration.

    Args:
        cfgs (dict): Configuration parameters, containing the type of learning rate scheduler to be used,
                     maximum learning rate, number of epochs, number of warmup steps and cycles.
        optimizer: The optimizer for which the learning rate scheduler will be applied.
        train_loader: The DataLoader object for the training data.

    Returns:
        lrs (torch.optim.lr_scheduler._LRScheduler): The specified learning rate scheduler.

    """
    if cfgs.lr_scheduler_name == "one_cycle_lr":
        lrs = lr_scheduler.OneCycleLR(optimizer,
                                      max_lr=cfgs.max_lr,
                                      epochs=cfgs.num_epochs,
                                      steps_per_epoch=len(train_loader))
    elif cfgs.lr_scheduler_name == "warmup_cosine_lr":
        lrs = WarmupCosineSchedule(optimizer,
                                   warmup_steps=cfgs.warmup_steps,
                                   total_steps=(cfgs.num_epochs * len(train_loader)),
                                   cycles=cfgs.cycles)
    elif cfgs.lr_scheduler_name == "warmup_sqrt_lr":
        lrs = WarmupSqrtDecayWithPercentageCoolDown(optimizer,
                                                    warmup_steps=cfgs.warmup_steps,
                                                    total_steps=(cfgs.num_epochs * len(train_loader)))
    else:
        lrs = None

    return lrs
