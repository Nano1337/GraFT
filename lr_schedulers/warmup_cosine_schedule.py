from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineSchedule(_LRScheduler):
    """ 
    Implements a learning rate scheduler with a warm-up phase and cosine decay. During the warm-up phase, 
    the learning rate linearly increases from 0 to 1 over `warmup_steps` training steps. Following the warm-up, 
    the learning rate decreases from 1 to 0 over the remaining `total_steps - warmup_steps` steps according to 
    a cosine curve. If `cycles` (default=0.5) is different from default, then the learning rate follows a cosine 
    function after the warm-up phase.

    Args:
        optimizer: The optimizer for which the learning rate scheduler will be applied.
        warmup_steps (int): The number of steps for the warmup phase.
        total_steps (int): The total number of steps of the training.
        cycles (float, optional): Determines the number of cosine cycles. Default is 0.5.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculates and returns the learning rate for the current training step.

        Returns:
            list of float: The learning rate for each parameter group.
        """
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cos_out = 0.5 * (1.0 + math.cos(math.pi * self.cycles * 2.0 * progress))
        
        return [base_lr * cos_out for base_lr in self.base_lrs]
