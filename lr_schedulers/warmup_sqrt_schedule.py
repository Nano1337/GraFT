from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupSqrtDecayWithPercentageCoolDown(_LRScheduler):
    """
    Implements a learning rate scheduler with a warm-up phase, square-root decay, and cool-down phase. 
    During the warm-up phase, the learning rate linearly increases from 0 to 1 over `warmup_steps` training steps. 
    Following the warm-up, the learning rate decays as max_lr * sqrt(w/step) for the next steps. 
    Finally, in the cool-down phase (which takes the last 10% of the total steps), 
    the learning rate linearly decays to 0 from the value at the end of the square-root decay phase.

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
        self.start_cool_down_lr = None  # This will store the LR at the start of the cool-down phase
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
                max(base_lr * self.last_epoch / self.warmup_steps, 0)
                for base_lr in self.base_lrs
            ]
        
        # Square-root decay phase
        elif self.last_epoch < self.total_steps - self.cooldown_steps:
            decay_factor = math.sqrt(self.warmup_steps / self.last_epoch)
            return [max(base_lr * decay_factor, 0) for base_lr in self.base_lrs]
        
        # Cool-down phase
        else:
            # At the start of the cool-down, store the current learning rate
            if self.start_cool_down_lr is None:
                self.start_cool_down_lr = [
                    base_lr * math.sqrt(self.warmup_steps / (self.total_steps - self.cooldown_steps))
                    for base_lr in self.base_lrs
                ]

            progress_cool_down = (self.last_epoch - (self.total_steps - self.cooldown_steps)) / self.cooldown_steps
            linear_decay_values = [
                max(start_lr * (1.0 - progress_cool_down), 0)
                for start_lr in self.start_cool_down_lr
            ]
            return linear_decay_values

    def reset(self, new_lr=None):
        """
        Resets the scheduler to its initial state.

        Args:
            new_lr (float, optional): If provided, resets the learning rate of the optimizer to this value.
                                    Otherwise, the learning rate is reset to its initial value.
        """
        # Reset last_epoch to its initial value
        self.last_epoch = -1
        # Reset the LR at the start of the cool-down phase
        self.start_cool_down_lr = None
        # Reset the optimizer's learning rate
        if new_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            for param_group, initial_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = initial_lr
