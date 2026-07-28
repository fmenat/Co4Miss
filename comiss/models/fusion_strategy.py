import torch
from torch import nn
from typing import List, Union, Dict

from .base_fusion import CoLearnMissing
from .fusion_module import FusionModuleMissing

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