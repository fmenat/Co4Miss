import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict
from torchmetrics import MetricCollection, F1Score

torch.set_float32_matmul_precision('high')

from .core_fusion import _BaseMissingViewsPL
from .co_loss import FeatureCoLoss, DecisionCoLoss
from .losses import get_loss_by_name
from .missing_utils import possible_missing_mask, augment_all_missing, augment_random_missing
from .single.base_decoders import Generic_Decoder
from .utils import stack_all, object_to_list, collate_all_list, detach_all

class CoLearnMissing(_BaseMissingViewsPL):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 predictive_core: nn.Module,
                 feat_fusion: nn.Module,
                 fusion_module: nn.Module,
                 n_labels: int,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 weights_loss: dict = {},
                 head_args: dict = {},
                 val_loss_flag: bool = False,
                 **kwargs,
                 ):
        super(CoLearnMissing, self).__init__(**kwargs)          
        self.view_names = list(self.view_encoders.keys()) if view_names == [] else view_names
        self.fusion_module = fusion_module
        self.feat_fusion = feat_fusion
        self.N_views = len(self.view_names)
        self.n_labels = n_labels
        self.weights_loss = weights_loss
        self.missing_as_aug = False
        self.val_loss_flag = val_loss_flag #whether to compute and log validation loss

        #for main prediction
        self.view_encoders = nn.ModuleDict(view_encoders)
        self.view_prediction_heads = {}
        for v_name ,model_v in view_encoders.items():
            emb_dim = model_v.get_output_size()
            self.view_prediction_heads[v_name] = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=self.n_labels, input_dim=emb_dim, **head_args)
        self.view_prediction_heads = nn.ModuleDict(self.view_prediction_heads)
        
        base_val_metrics = MetricCollection({
            "f1ma": F1Score(num_classes=self.n_labels, task="multiclass", average='macro'),
            "f1we": F1Score(num_classes=self.n_labels, task="multiclass", average='weighted'),
            }).clone(prefix="va_")
        self.val_miss_metrics = base_val_metrics.clone(postfix="_miss")
        self.val_full_metrics = base_val_metrics.clone(postfix="_full")

        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        modality_head_awr_inf = Generic_Decoder(copy.deepcopy(predictive_core), out_dims=len(self.view_names), input_dim=emb_dim//2)
        self.feature_coloss = FeatureCoLoss(shared_loss=self.weights_loss.get("shared", [])[0], 
                                            specific_loss=self.weights_loss.get("specific", [])[0], 
                                            contrastive_temp=self.weights_loss.get("contrastive_temp", 0.1),
                                            discriminator = modality_head_awr_inf)
        self.decision_coloss = DecisionCoLoss(distillation_loss=self.weights_loss.get("decision", [])[0],
                                              distillation_temp=self.weights_loss.get("distillation_temp", 1),
                                              distillation_weight=self.weights_loss.get("distillation_weight", 1))
    
    def validation_step(self, batch: dict, batch_idx):
        loss, preds_val = self.loss_batch(batch)
        self.val_miss_metrics.update(preds_val["pred:missing"].detach(), preds_val["target:missing"].detach())
        self.val_full_metrics.update(preds_val["pred:full"].detach(), preds_val["target:full"].detach())
        if self.val_loss_flag:
            for k, v in loss.items():
                self.log("va_" + k, v, prog_bar=True)
            return loss["objective"]
    
    def on_validation_epoch_end(self):
        dic_metrics_miss = self.val_miss_metrics.compute()
        dic_metrics_full = self.val_full_metrics.compute()
        dic_metrics = {**dic_metrics_miss, **dic_metrics_full}
        for k in dic_metrics_miss:
            m_name = k.split("_miss")[0]
            dic_metrics[m_name+"_avg"] = (dic_metrics_miss[m_name+"_miss"] + dic_metrics_full[m_name+"_full"])/2

        for k, v in dic_metrics.items():
            self.log(f"{k}", v, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.val_miss_metrics.reset()
        self.val_full_metrics.reset()

    def set_missing_info(self, aug_status, missing_random: bool=False, random_perc = 0,**kwargs):
        #set the status of the missing as augmentation technique during training
        self.missing_as_aug = aug_status
        self.missing_random = missing_random
        self.random_perc = random_perc
        self.missing_method={}
        self.all_missing_views = augment_all_missing(self.view_names, skip_full=True)
        print("The missing as augmentation options is", possible_missing_mask(len(self.view_names)))

    def forward_encoders(self,views: Dict[str, torch.Tensor], inference_views: list = []) -> Dict[str, torch.Tensor]:
        inference_views = self.view_names if len(inference_views) == 0 else inference_views
        zs_views = {}
        for v_name in self.view_names:
            if v_name in inference_views and v_name in views:
                zs_views[v_name] = self.view_encoders[v_name](views[v_name])  
        return zs_views

    def forward(self, 
                views: Dict[str, torch.Tensor], 
                out_norm:str = "", 
                intermediate:bool=True, 
                inference_views: list=[],
                **kwargs):
        out_zs_views = self.forward_encoders(views, inference_views=inference_views)
                
        perview_out_pred = {}
        aux_reps = {"specific":{}, "shared":{}}
        for v_name, s_full in out_zs_views.items():
            nfeat = s_full.shape[1]//2
            aux_reps["specific"][v_name] = s_full[:, :nfeat]
            aux_reps["shared"][v_name]   =  s_full[:, nfeat:]

            #main prediction for L_CL -
            perview_out_pred[v_name] = self.view_prediction_heads[v_name]( self.feat_fusion([aux_reps["specific"][v_name], aux_reps["shared"][v_name]], 
                                                             views_available=[])["joint_rep"] ) 
            
        #fusion -- hardcoded to get average over available data
        if len(inference_views) == 0:
            views_available_ohv = torch.ones(self.N_views)
            perview_pred_forced = list(perview_out_pred.values())
        else:
            views_available_ohv = []
            perview_pred_forced = []
            for v in self.view_names:
                if v in perview_out_pred:
                    views_available_ohv.append(1)
                    perview_pred_forced.append(perview_out_pred[v])
                else:
                    views_available_ohv.append(0)
                    perview_pred_forced.append(torch.zeros_like(perview_out_pred[v_name]))
            views_available_ohv = torch.Tensor(views_available_ohv)

        main_out_pred = self.fusion_module(perview_pred_forced, views_available=views_available_ohv.bool())["joint_rep"]
        
        return_dict = {"prediction": self.apply_norm_out(main_out_pred, out_norm)} #normalize output (softmax/sigmoid) when is used in inference
        if intermediate:
            return_dict["views:prediction"] = perview_out_pred
            return_dict["views:rep"] = aux_reps
        return return_dict    
    
    def loss_batch(self, batch: dict):
        views_dict, target_all = self.prepare_batch(batch)
        
        out_dic_full = self(views_dict) #full-data prediction only once

        ## MISSING DATA PART
        if self.missing_as_aug or not self.training: #MAUG at decision -level (only play with prediction vectors)
            if self.missing_random: #force random for many views
                self.all_missing_views = augment_random_missing(self.view_names, perc=self.random_perc, skip_full=True)

            views_targets = []
            fused_preds_missing = [] #for mutual distillation
            for i, inference_views in enumerate(self.all_missing_views): #a bit slow if it is random
                views_available_ohv = torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
                perview_pred_masked = {v: out_dic_full["views:prediction"][v] for v in inference_views}

                perview_pred_forced = [perview_pred_masked[v] if v in perview_pred_masked else torch.zeros_like(list(perview_pred_masked.values())[0]) for v in self.view_names]
                out_dic_missing_ = {'prediction': self.fusion_module(perview_pred_forced, views_available=views_available_ohv.bool())["joint_rep"] }                
                if i == 0:
                    out_dic_missing = object_to_list(out_dic_missing_) #to start append values
                else:
                    collate_all_list(out_dic_missing, out_dic_missing_) #add to list in cpu
                views_targets.append(target_all)
                fused_preds_missing.append(out_dic_full["prediction"])
            out_dic_missing = stack_all(out_dic_missing, data_type="torch")
            missing_pred = out_dic_missing["prediction"] #prediction with missing data
            views_targets = torch.concat(views_targets, axis = 0)
            fused_preds_missing = torch.concat(fused_preds_missing, dim=0)
        
        ## LOSS PART
        fused_pred = out_dic_full["prediction"]
        pred_view = out_dic_full["views:prediction"]
        rep_view = out_dic_full["views:rep"]

        if not self.training:
            preds_val = {"pred:full": fused_pred,
                     "pred:missing": torch.concat([missing_pred], axis=0), 
                     "target:full":target_all,
                      "target:missing": torch.concat([views_targets], axis=0)}
            if not self.val_loss_flag:
                return None, preds_val
        
        loss_dic = {}
        
        #fused loss
        loss_dic["Main"] =  self.criteria(fused_pred, target_all) 
        loss_full = self.weights_loss.get("main_pred",1)*loss_dic["Main"]

        ####### Feature-level co-learning #######
        loss_feat = 0
        loss_dic_feature = self.feature_coloss(list(rep_view["shared"].values()), list(rep_view["specific"].values()))
        for loss_type, loss_info in self.weights_loss.items():
            if loss_type in ["shared", "specific"]:
                loss_name, loss_value = loss_info
                loss_dic[loss_name] = loss_dic_feature[f"{loss_name}"]
                loss_feat += loss_value*loss_dic[loss_name]

        #### Decision-level co-learning ####
        loss_dic_decision = self.decision_coloss(list(pred_view.values()), 
                                                 fused_pred if self.decision_coloss.distillation_loss == "mutual_distillation" else fused_preds_missing, 
                                                 target_all if self.decision_coloss.distillation_loss == "mutual_distillation" else views_targets,
                                                 missing_pred if self.decision_coloss.distillation_loss == "missing_distillation" else None)
        for loss_type, loss_info in self.weights_loss.items():
            if loss_type == "decision":
                loss_name, loss_value = loss_info
                loss_dic[loss_name] = loss_dic_decision[f"{loss_name}"]
                loss_dec = loss_value*loss_dic[loss_name]


        total_loss = loss_full +  loss_feat + loss_dec
        if self.training:
            return {"objective": total_loss, **loss_dic}
        else:
            return {"objective": total_loss, **loss_dic}, preds_val
    

    def get_encoder_models(self):
        return self.view_encoders
    
    def get_perview_model(self, view_name):
        return nn.ModuleList([self.view_encoders[view_name], self.view_prediction_heads[view_name]])
    
    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate: bool=True,
            out_norm:str="",
            device:str="",
            args_forward:dict = {},
            perc_forward: float = 1,
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        missing_forward = False #flag to use random forward (based on percentage) in testing cases
        self.eval() #set batchnorm and dropout off
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, _ = self.prepare_batch(batch)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)

                if perc_forward == 1:
                    missing_forward = True
                elif perc_forward !=0: 
                    if np.random.rand() < perc_forward: #forward with missing 
                        missing_forward = True
                    
                if missing_forward:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm, **args_forward)
                    missing_forward = False
                else:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm)
                
                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) #to start append values
                else:
                    collate_all_list(outputs, outputs_) #add to list in cpu
        self.train()
        return stack_all(outputs) #stack with numpy in cpu
    