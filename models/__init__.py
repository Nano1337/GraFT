from typing import Optional, Any

import torch
import torch.distributed as dist

from models.DeiT_gradual_fusion import DEIT_Gradual_Fusion
from models.DeiT_gradual_fusion_scalable import DEIT_Gradual_Fusion_Scalable
from models.DINO_gradual_fusion import DINO_Gradual_Fusion

from models.DeiT_vanilla_fusion import DEIT_Vanilla_Fusion
from models.DeiT_gradual_fusion_v2 import DEIT_Gradual_Fusion_V2

def get_model(cfgs: dict, fabric: any) -> torch.nn.Module:
    """Gets the specified model based on the provided configurations.

    Args:
        cfgs: The configuration parameters for the model.
        fabric: The computational device to use for calculations.

    Returns:
        The specified model.

    Raises:
        NotImplementedError: If the model specified in the configurations is not found.
    """
    if cfgs.model_name == "deit_gradual_fusion":
        model = DEIT_Gradual_Fusion(cfg=cfgs, fabric=fabric)
    elif cfgs.model_name == "dino_gradual_fusion":
        model = DINO_Gradual_Fusion(cfg=cfgs, fabric=fabric)
    elif cfgs.model_name == "deit_gradual_fusion_scalable":
        model = DEIT_Gradual_Fusion_Scalable(cfg=cfgs, fabric=fabric)
    elif cfgs.model_name == "deit_vanilla_fusion":
        model = DEIT_Vanilla_Fusion(cfg=cfgs, fabric=fabric)
    elif cfgs.model_name == "deit_gradual_fusion_v2":
        model = DEIT_Gradual_Fusion_V2(cfg=cfgs, fabric=fabric)
    else:
        model = None

    if model is None:
        raise NotImplementedError(f"{cfgs.model_name} not found")

    return model
