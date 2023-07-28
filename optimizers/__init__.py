import torch.optim as optim

def get_optim(cfgs: dict, model): 
    """
    This function returns an optimizer for the provided model based on the specified configuration parameters.

    Args:
        cfgs (dict): A dictionary containing the configuration parameters for the optimizer. This dictionary should 
                     contain the following keys: 
                        'optimizer' - The type of optimizer to use. It could be 'adam' or 'adamw'.
                        'lr' - The learning rate to be used for the optimizer.
                        'weight_decay' - (Only for 'adamw') The weight decay factor.
                        'beta1', 'beta2' - (Only for 'adamw') The coefficients used for computing running averages 
                                           of gradient and its square.
        model (nn.Module): The PyTorch model for which the optimizer will be created.

    Raises:
        NotImplementedError: If the optimizer specified in 'cfgs' is neither 'adam' nor 'adamw'.

    Returns:
        torch.optim.Optimizer: The optimizer for the provided model based on the configuration parameters.
    """
    if cfgs.optimizer == "adam": 
        optimizer = optim.Adam(model.parameters(), lr=cfgs.lr)
    elif cfgs.optimizer == "adamw": 
        optimizer = optim.AdamW(model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay, betas=(cfgs.beta1, cfgs.beta2))
    else: 
        raise NotImplementedError(f"{cfgs.optimizer} not found")
    return optimizer