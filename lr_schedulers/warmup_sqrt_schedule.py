from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupSqrtDecayWithPercentageCoolDown(_LRScheduler):
    """
    Implements a learning rate scheduler with a warm-up phase, square-root decay, and cool-down phase. 
    During the warm-up phase, the learning rate linearly increases from 0 to 1 over `warmup_steps` training steps. 
    Following the warm-up, the learning rate decays as max_lr * sqrt(w/step) for the next steps. 
    Finally, in the cool-down phase (which takes the last 10% of the total steps), 
    the learning rate linearly decays to 0.

    Args:
        optimizer: The optimizer for which the learning rate scheduler will be applied.
        warmup_steps (int): The number of steps for the warmup phase.
        total_steps (int): The total number of steps of the training.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
    """

    def __init__(self,
                 optimizer,
                 warmup_steps,
                 total_steps,
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cooldown_steps = int(0.1 * total_steps)
        super(WarmupSqrtDecayWithPercentageCoolDown, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculates and returns the learning rate for the current training step.

        Returns:
            list of float: The learning rate for each parameter group.
        """
        # Warm-up phase
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        
        # Square-root decay phase
        elif self.last_epoch < self.total_steps - self.cooldown_steps:
            decay_factor = math.sqrt(self.warmup_steps / self.last_epoch)
            return [base_lr * decay_factor for base_lr in self.base_lrs]
        
        # Cool-down phase
        else:
            progress_cool_down = (self.last_epoch - (self.total_steps - self.cooldown_steps)) / self.cooldown_steps
            linear_decay = 1.0 - progress_cool_down
            return [base_lr * linear_decay for base_lr in self.base_lrs]
