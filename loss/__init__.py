import torch.nn as nn
from loss.reid_loss import Triplet_CE_Loss, Combined_Loss


def get_loss(cfgs: dict, fabric):
    """
    Function to get the specified loss function according to the configuration.

    Args:
        cfgs (dict): Configuration parameters containing the type of loss function to be used.
        fabric: An object which holds methods for distributed training.

    Returns:
        criterion (nn.Module): The specified PyTorch loss function.

    Raises:
        NotImplementedError: If the specified loss function in the config is not supported.
    """
    if cfgs.loss_fn == "CrossEntropy":
        criterion = nn.CrossEntropyLoss(label_smoothing=cfgs.smoothing)
    elif cfgs.loss_fn == "MSE":
        criterion = nn.MSELoss()
    elif "triplet" in cfgs.loss_fn and "+" not in cfgs.loss_fn:
        print("Using triplet_CE loss function.")
        criterion = Triplet_CE_Loss(cfgs=cfgs, fabric=fabric)
    elif "+" in cfgs.loss_fn:
        print("Using combined loss function.")
        criterion = Combined_Loss(cfgs=cfgs, fabric=fabric)
    else:
        raise NotImplementedError(
            "The specified loss function is not supported.")
    return criterion
