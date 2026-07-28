import torch
from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusionMissing, SVPool, SVEncoders, SingleView_Student, ShaSpec, CoLearnMissing, DModDrop,Feat4Miss
from .fusion_module import FusionModuleMissing

class FeatureFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 **kwargs,
                 ):
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders","fusion_module", "predictive_model"])

class DecisionFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: dict = {},
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 n_outputs: int = 0,
                 **kwargs,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_encoders))]}
            fusion_module = FusionModuleMissing(**fusion_module)
        super(DecisionFusion, self).__init__(view_encoders, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names, **kwargs)
        self.n_outputs = n_outputs
        self.save_hyperparameters(ignore=["view_encoders","fusion_module"])


class SingleViewPool(SVPool):
    pass

class ShaSpecMV(ShaSpec):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 shared_view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 loss_args: dict ={},
                 view_names: List[str] = [], #this is only used if view_encoders are a list
                 **kwargs,
                 ):
        super(ShaSpecMV, self).__init__(view_encoders, fusion_module, prediction_head, shared_view_encoders,
                                         loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders","fusion_module","prediction_head", "shared_view_encoders"])

class DModDropMV(DModDrop):
    def __init__(self,
                 view_models,
                 fusion_module: dict = {},
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 n_outputs: int = 0,
                 **kwargs,
                 ):
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "avg", "adaptive":False, "emb_dims":[n_outputs for _ in range(len(view_models))]}
            fusion_module = FusionModuleMissing(**fusion_module)
        super(DModDropMV, self).__init__(view_models, fusion_module, loss_args=loss_args, view_names=view_names, **kwargs)
        self.n_outputs = n_outputs
        self.save_hyperparameters(ignore=["view_encoders","fusion_module"]) 


class Feature4Missing(Feat4Miss):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 **kwargs,
                 ):
        super(Feature4Missing, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders","fusion_module", "predictive_model"])

class CoLearnMiss(CoLearnMissing):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 predictive_core: nn.Module,
                 fusion_module: nn.Module,
                 n_labels: int,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 **kwargs,
                 ):
        feat_fusion = FusionModuleMissing(**{"mode": "concat", "emb_dims": [list(view_encoders.values())[0].get_output_size()//2]*2 })
        
        super(CoLearnMiss, self).__init__(view_encoders, predictive_core, feat_fusion, fusion_module, n_labels, 
                                          loss_args=loss_args, view_names=view_names, **kwargs)
        self.save_hyperparameters(ignore=["view_encoders","predictive_core","fusion_module", "feat_fusion"])