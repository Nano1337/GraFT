
from models.DeiT_gradual_fusion import DEIT_Gradual_Fusion

def get_model(cfgs: dict, fabric: any):

    if cfgs.model_name == "deit_gradual_fusion":
        model = DEIT_Gradual_Fusion(cfg=cfgs, fabric=fabric)
    else: 
        model = None

    # Check if the model was found
    if model is None:
        raise NotImplementedError(f"{cfgs.model_name} not found")

    return model
    

