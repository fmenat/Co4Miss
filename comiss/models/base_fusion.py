import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict
import math
from torchmetrics import MetricCollection, Precision, Recall, F1Score

torch.set_float32_matmul_precision('high')

from .core_fusion import _BaseMissingViewsPL
from .co_loss import FeatureCoLoss, DecisionCoLoss
from .fusion_module import FusionModuleMissing
from .losses import get_loss_by_name, contrastive_loss, cross_bt_loss, single_bt_loss, pairwise_contrastive_loss, mutual_distillation_loss, supervised_contrastive_loss, volume_computation
from .missing_utils import possible_missing_mask, augment_all_missing, augment_random_missing
from .nn_models import create_model
from .single.base_decoders import Generic_Decoder
from .utils import stack_all, object_to_list, collate_all_list, detach_all, map_encoders


class MVFusionMissing(_BaseMissingViewsPL):
    #ONLY FOR POINT-PREDICTION
    #it is based on three modules: encoders, aggregation, prediction_head
    #only one-task (prediction) and full-view available setting
    #support list and dictionary of encoders -- but transform to dict (ideally it should be always a dict)
    #support list and dictionary of emb dims -- but transform to dict
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],  #require that it contain get_output_size() .. otherwise indicate in emb_dims..
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], #this is only used if view_encoders are a list
                 **kwargs,
                 ):
        super(MVFusionMissing, self).__init__(**kwargs)
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")

        view_encoders = map_encoders(view_encoders, view_names=view_names) #view_encoders to dict if no dict yet (e.g. list)
        self.view_encoders = nn.ModuleDict(view_encoders)
        self.view_names = list(self.view_encoders.keys())
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head
        self.N_views = len(self.view_encoders)
        
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.missing_as_aug=False            

    def set_missing_info(self, aug_status, name:str="impute", where:str ="", value_fill=None, missing_random: bool=False, random_perc = 0,**kwargs):
        #set the status of the missing as augmentation technique during training
        self.missing_as_aug = aug_status
        if name =="impute": #for case of impute
            where = "input" if where == "" else where #default value: input
            value_fill = 0.0 if type(value_fill) == type(None) else value_fill #default value: 0.0
        elif name == "adapt": #for case of adapt
            where = "feature" if where =="" else where #default value: input
            value_fill = torch.nan if type(value_fill) == type(None) else value_fill #default value: 0.0
        elif name == "ignore": #completly ignore/drop missing data
            pass
        self.missing_method = {"name": name, "where": where, "value_fill": value_fill}
        self.missing_random = missing_random
        self.random_perc = random_perc
        if self.missing_as_aug:
            self.all_missing_views = augment_all_missing(self.view_names)
            print("The missing as augmentation was set on with an augmentation of ", possible_missing_mask(len(self.view_names)),"and following missing_method =",self.missing_method) #len(self.view_encoders.keys())

    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            inference_views: list = [],
            missing_method: dict = {},
            ) -> Dict[str, torch.Tensor]:
        inference_views = self.view_names if len(inference_views) == 0 else inference_views

        zs_views = {}
        for v_name in self.view_names:
            forward_f = True  #just a flag when no forward for missing views

            if v_name in inference_views and v_name in views:
                data_forward = views[v_name]

            else: 
                if missing_method.get("where") == "input": #fill when view not in testing forward or view is missing
                    data_forward = torch.ones_like(views[v_name])*missing_method["value_fill"] #asumming data is available

                elif missing_method.get("where") == "feature": #avoid forward and fill at feature
                    forward_f = False
                    value_fill = torch.nan if missing_method["value_fill"] == "nan" else missing_method["value_fill"]
                    zs_views[v_name] = torch.ones(self.view_encoders[v_name].get_output_size(), device=self.device).repeat(
                                                                                                        list(views.values())[0].shape[0], 1)*value_fill
                
                elif missing_method.get("name") == "ignore": #do not forward over missing views -- just remove and keep adapt?
                    forward_f=False
                    
                else:
                    raise Exception("Inference with few number of views (missing) but no missing method *where* was indicated in the arguments")

            if forward_f:
                zs_views[v_name] = self.view_encoders[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate:bool = True,
            out_norm:str="",
            inference_views: list = [],
            missing_method: dict = {}, 
            forward_from_representation: bool = False, #for only predictions based on pre-created features
            ) -> Dict[str, torch.Tensor]:
        #encoders        
        if forward_from_representation:
            out_zs_views = {"views:rep": views}
        else:
            out_zs_views = self.forward_encoders(views, inference_views=inference_views, missing_method=missing_method) 
        
        #merge function
        if len(inference_views) != 0 and missing_method.get("name") == "ignore":  #this is usually used with MAUG technique
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names if v in inference_views] 
        else: #adapt, impute or others
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names] # this ensures that the same views are passed for training
        views_available_ohv = torch.ones(self.N_views) if len(inference_views) == 0 else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        out_z_e = self.fusion_module(views_data, views_available=views_available_ohv.bool())
        
        #prediction head
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return_dic = {"prediction": self.apply_norm_out(out_y, out_norm)}
        if intermediate:
            return_dic["last_layer"] = out_y
            return dict( **return_dic, **out_zs_views, **out_z_e)
        else:
            return return_dic
    
    def loss_batch(self, batch: dict) -> dict:
        #calculate values of loss that will not return the model in a forward/transform
        views_dict, views_target = self.prepare_batch(batch)
        if self.missing_as_aug and self.training:
            if self.missing_random: #a single random augmentation instance -- remove in published version, only CoM
                missing_case = augment_random_missing(self.view_names, perc=self.random_perc) #return a list of augmentations
                out_dic = self(views_dict, inference_views=missing_case[0], missing_method=self.missing_method)
                views_targets = views_target
            else:
                    
                zs_views = self.forward_encoders(views_dict)["views:rep"]

                views_targets = []
                for i, inference_views in enumerate(self.all_missing_views):
                    if self.missing_method.get("name") == "ignore":
                        aux_zs_views = {k: v for k,v in zs_views.items()} #to force copy dictionary
                    else:
                        value_fill = torch.nan if self.missing_method["value_fill"] == "nan" else self.missing_method["value_fill"] 
                        aux_zs_views = {k: (v if k in inference_views else torch.ones_like(v)*value_fill) for k, v in zs_views.items()}
                        
                    out_dic_ = self(aux_zs_views, inference_views=inference_views, missing_method=self.missing_method, forward_from_representation = True)

                    if i == 0:
                        out_dic = object_to_list(out_dic_) #to start append values
                    else:
                        collate_all_list(out_dic, out_dic_) #add to list in cpu
                    views_targets.append(views_target)
                out_dic = stack_all(out_dic, data_type="torch")
                views_targets = torch.concat(views_targets, axis = 0)
        else: #in case self.missing_as_aug = True but not in training the model will calculate loss without missing views
            out_dic = self(views_dict) 
            views_targets = views_target

        if self.hparams_initial.get("focus_full_view") and (len(views_target) != len(views_targets)): 
            #doble condition to check if missing as aug is used, then focus on the full-view predictions
            #in case self.missing_as_aug = True but not in training the model will calculate loss without missing views (else condition)
            return_dic =  {"objective": self.criteria(out_dic["prediction"][:len(views_target)], views_target) +
                    self.criteria(out_dic["prediction"][len(views_target):], views_targets[len(views_target):])}
        else: #default function for most of the methods
            return_dic = {"objective": self.criteria(out_dic["prediction"], views_targets)}

        if self.missing_as_aug and self.training and not self.missing_random: #multiple missing views cases
            return_dic["fullview"] =  self.criteria(out_dic["prediction"][:len(views_target)], views_target)
            return_dic["missingview"] =  self.criteria(out_dic["prediction"][len(views_target):], views_targets[len(views_target):])
        return return_dic    
            
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
    
class SVPool(MVFusionMissing):
    #train single-models per view in a pool setting
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 view_prediction_heads: Union[List[nn.Module],Dict[str,nn.Module]],
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 orthonogalseq_flag: bool=False, #whether to active orthonogal directions in their gradients
                 cotraining_flag: bool = False, #whether to use co-training at decision lvl
                 cotraining_weight: float = 1,
                 **kwargs,
                 ):
        super(SVPool, self).__init__(view_encoders, nn.Identity(), nn.Identity(),
            loss_args=loss_args, view_names=view_names, **kwargs)    
        
        self.view_prediction_heads = nn.ModuleDict(view_prediction_heads)
        self.prediction_aggregation = FusionModuleMissing([v.get_output_size() for v in self.view_prediction_heads.values()], mode="avg_ignore") #average by ignoring missing predictions
        self.orthonogalseq_flag = orthonogalseq_flag
        if self.orthonogalseq_flag:
            with torch.no_grad():
                self.Pl = torch.autograd.Variable(torch.eye(list(self.view_encoders.values())[0].get_output_size()).type(torch.float32))
        self.cotraining_flag = cotraining_flag
        if self.cotraining_flag:
            self.decision_criteria = nn.KLDivLoss(reduction='batchmean', log_target=True)
            self.cotraining_weight = cotraining_weight

    def forward(self, views: Dict[str, torch.Tensor], out_norm:str = "", inference_views: list = [], intermediate=True, **kwargs):
        out_zs_views = self.forward_encoders(views)
                
        out_y_zs = {}
        for v_name in out_zs_views["views:rep"]:    
            if len(inference_views) != 0 : #batch-wise masking in average ensemble
                if v_name in inference_views: 
                    out_y = self.view_prediction_heads[v_name]( out_zs_views["views:rep"][v_name])
                else:
                    out_y  = torch.ones(len(out_zs_views["views:rep"][v_name]), self.view_prediction_heads[v_name].get_output_size(), device=self.device)*torch.nan
            else:
                out_y = self.view_prediction_heads[v_name]( out_zs_views["views:rep"][v_name])
            out_y_zs[v_name] = self.apply_norm_out(out_y, out_norm)
        aggregated_pred = self.prediction_aggregation(list(out_y_zs.values()), views_available=[])["joint_rep"]
    
        if intermediate:
            return dict(out_zs_views, **{"views:prediction": out_y_zs, "prediction": aggregated_pred})
        else:
            return {"views:prediction": out_y_zs, "prediction": aggregated_pred}

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        yi_xi = out_dic["views:prediction"]

        self.current_view_names = list(yi_xi.keys())
        loss_dic = { }
        loss_aux = 0
        if self.orthonogalseq_flag:
            t = self.global_step % len(self.current_view_names)
            v_name = self.current_view_names[t]
            loss_dic["Lo"+v_name] = self.criteria(yi_xi[v_name], views_target)
            loss_aux += loss_dic["Lo"+v_name]
            self.current_out = out_dic["views:rep"][v_name]
        elif self.cotraining_flag:
            total_loss_co = 0
            for v_name in self.current_view_names:
                loss_dic["Lo"+v_name] = self.criteria(yi_xi[v_name], views_target)
                loss_aux += loss_dic["Lo"+v_name]

                yi_xi_for_loss = nn.functional.log_softmax(yi_xi[v_name], dim=-1)
                
                for v_name2 in self.current_view_names:
                    if v_name2 == v_name:
                        continue

                    yi_xi_for_loss_aux = nn.functional.log_softmax(yi_xi[v_name2], dim=-1).detach()
                    total_loss_co += self.decision_criteria(yi_xi_for_loss, yi_xi_for_loss_aux)

            loss_dic["LoCotr"] = total_loss_co/(len(self.current_view_names)-1)
            loss_aux += self.cotraining_weight*loss_dic["LoCotr"]
        else:  #default
            for v_name in self.current_view_names:
                loss_dic["Lo"+v_name] = self.criteria(yi_xi[v_name], views_target)
                loss_aux += loss_dic["Lo"+v_name]

        return {"objective": loss_aux, **loss_dic}
    
    def get_encoder_models(self):
        return self.view_encoders

    def get_sv_models(self):
        return self.view_prediction_heads
    
    def set_aggregation(self, mode="avg_ignore"):
        self.prediction_aggregation = FusionModuleMissing([v.get_output_size() for v in self.view_prediction_heads.values()], mode=mode) 
    
    def on_after_backward(self):
        if self.orthonogalseq_flag:
            #from https://github.com/Cecile-hi/Multimodal-Learning-with-Alternating-Unimodal-Adaptation
            t = self.global_step % len(self.current_view_names)
            for n, w in self.view_prediction_heads[self.current_view_names[t]].named_parameters():
                if n.endswith(".weight"):
                    self.Pl = self.Pl.to(w.device)
                    r = torch.mean(self.current_out, 0, keepdim=True)
                    k = torch.mm(self.Pl, torch.t(r))
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (1e-7 + torch.mm(k, r)))

                    pnorm2 = torch.linalg.norm(self.Pl.data, ord='fro')

                    self.Pl.data = self.Pl.data / pnorm2
                    if len(w.grad.data.shape) == 1:
                        w.grad.data = torch.mm(w.grad.data.unsqueeze(0), torch.t(self.Pl.data)).squeeze()
                    else:
                        w.grad.data = torch.mm(w.grad.data, torch.t(self.Pl.data))


class SingleView_Student(MVFusionMissing):
    def __init__(self,
                 teacher_model,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]], 
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 weights_kd: dict = {},
                 **kwargs
                 ):
        super(SingleView_Student, self).__init__(view_encoders,fusion_module,prediction_head,loss_args, **kwargs)
        
        self.teacher_model = copy.deepcopy(teacher_model)
        self.teacher_model.head = self.teacher_model.prediction_head[1]
        self.teacher_model.prediction_head = self.teacher_model.prediction_head[0]
        self.teacher_model.eval()

        self.criteria_kd = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.criteria_feat = nn.MSELoss(reduction="mean")
        if len(weights_kd) != 0:
            self.weights_kd = weights_kd
        else:
            self.weights_kd = {"main": 1, "feature": 1, "decision": 1}
        
    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        student_out = self(views_dict) 
        
        loss_main = self.criteria(student_out["prediction"], views_target)
        loss_aux = 0
        loss_dic = {}
        
        with torch.no_grad():
            teacher_out_feat = self.teacher_model(views_dict)["prediction"]
            teacher_out_pred = self.teacher_model.head(teacher_out_feat)

        student_out_pred = nn.functional.log_softmax(student_out["prediction"], dim=-1)
        teacher_out_pred = nn.functional.log_softmax(teacher_out_pred, dim=-1)        
        loss_dic["kd_dec"] = self.criteria_kd(student_out_pred, teacher_out_pred)     
        loss_aux += self.weights_kd["decision"]*loss_dic["kd_dec"]

        if len(teacher_out_pred.shape) > 1:
            class_p_teacher = teacher_out_pred.argmax(axis=-1)
            class_p_student = student_out["prediction"].argmax(axis=-1) 

            classes_batch = list( set(class_p_teacher.tolist()) & set(class_p_student.tolist()) )
            if len(classes_batch) == 0:
                loss_dic["kd_feat"] = 0
            else:
                total_loss_classes = 0
                for n_c in classes_batch:
                    protfeat_nc_student = student_out["joint_rep"][class_p_teacher == n_c].mean(axis=0)
                    protfeat_nc_teacher = teacher_out_feat[class_p_student == n_c].mean(axis=0)
                    total_loss_classes += self.criteria_feat(protfeat_nc_student, protfeat_nc_teacher)
                loss_dic["kd_feat"] = total_loss_classes/len(classes_batch)
        else:
            loss_dic["kd_feat"] = self.criteria_feat(student_out["joint_rep"], teacher_out_feat)
        loss_aux += self.weights_kd["feature"]*loss_dic["kd_feat"]

        return {"objective": self.weights_kd["main"]*loss_main+loss_aux,"lossmain":loss_main, **loss_dic} 
    

class SVEncoders(MVFusionMissing):
    #train view-dedicated encoders in a pool setting
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 view_names: List[str] = [],
                 loss_type: str = "contrastive",
                 loss_args: dict ={},
                 **kwargs,
                 ):
        super(SVEncoders, self).__init__(view_encoders, nn.Identity(), nn.Identity(),
            loss_args={"name": "mse"}, view_names=view_names, **kwargs)  
        self.loss_type = loss_type  

        if self.loss_type == "mse": #from Decoupling Features for Remote Sensing Missing Modality Learning
            self.cross_loss = nn.MSELoss(reduction="mean")
       
        elif self.loss_type == "contrastive": #from Omnisat/SkySense
            self.cross_loss = contrastive_loss(tau=loss_args.get("tau", 0.1), normalize=loss_args.get("normalize", True))
        
        elif self.loss_type == "cross": #from Decoupling Common and Unique Representations for Multimodal Self-supervised Learning
            self.dim_common = loss_args.get("dim_common", list(self.view_encoders.values())[0].get_output_size()/2)
            if loss_args.get("normalize", True):
                self.bn = nn.BatchNorm1d(list(self.view_encoders.values())[0].get_output_size(), affine=False)
            else:
                self.bn = nn.Identity()
            self.cross_loss = cross_bt_loss(dim_common=self.dim_common,norm_func=self.bn, lambda_=loss_args.get("lambda_", 1))
            self.single_loss = single_bt_loss(norm_func=self.bn, lambda_=loss_args.get("lambda_", 1))
        
        elif self.loss_type == "triplet":
            raise Exception("Triplet loss not implemented yet")
        
    def forward(self, views: Dict[str, torch.Tensor], **kwargs):
        return self.forward_encoders(views)

    def loss_batch(self, batch: dict):
        views_dict, _ = self.prepare_batch(batch)
        out_dic = self(views_dict)

        if self.loss_type == "cross":
            views_dict_aug = {k: views_dict[k]  + torch.empty_like(views_dict[k]).normal_(mean=0,std=0.1) for k in views_dict}   
            out_dic_aug = self(views_dict_aug)

        loss_main = 0
        loss_dic = {"comm": 0, "uniq": 0} if self.loss_type == "cross" else {}
        n_combs = 0
        for i, v_name in enumerate(out_dic["views:rep"]):
            if self.loss_type == "cross": 
                loss_dic[f"ind_{v_name}"] = self.single_loss(out_dic["views:rep"][v_name], out_dic_aug["views:rep"][v_name])["loss"]

            for j in range(len(out_dic["views:rep"])):
                if i == j:
                    continue
                v_name2 = list(out_dic["views:rep"].keys())[j]
                n_combs+=1
                
                if self.loss_type in ["mse", "contrastive"]: 
                    loss_main += self.cross_loss(out_dic["views:rep"][v_name], out_dic["views:rep"][v_name2])
                elif self.loss_type == "cross": 
                    loss_d = self.cross_loss(out_dic["views:rep"][v_name], out_dic["views:rep"][v_name2])
                    loss_dic["comm"] += loss_d["common"]
                    loss_dic["uniq"] += loss_d["unique"]
                    
        if self.loss_type == "cross": 
            loss_main = (loss_dic["comm"] + loss_dic["uniq"])/2 + sum([loss_dic[v] for v in loss_dic if "ind_" in v])
            loss_dic["comm"] = loss_dic["comm"]/n_combs
            loss_dic["uniq"] = loss_dic["uniq"]/n_combs
        return {"objective": loss_main/(len(self.view_names)+n_combs), **loss_dic}
    
    def get_encoder_models(self):
        return self.view_encoders
    

class ShaSpec(MVFusionMissing):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],  #require that it contain get_output_size() .. otherwise indicate in emb_dims..
                 fusion_module: nn.Module,
                 prediction_head: nn.Module, 
                 shared_view_encoders: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], 
                 weights_loss = {},
                 **kwargs,
                 ):
        super(ShaSpec, self).__init__(view_encoders, fusion_module, prediction_head,
            loss_args=loss_args, view_names=view_names, **kwargs)          
        self.view_encoders = None
        self.weights_loss = weights_loss
        
        #for encoding specific and shared information
        self.view_encoders_specific = {}
        self.view_encoders_shared = {}
        for v_name  in view_encoders:
            self.view_encoders_specific[v_name] = copy.deepcopy(view_encoders[v_name])
            self.view_encoders_shared[v_name] = nn.Sequential(copy.deepcopy(view_encoders[v_name]), shared_view_encoders) 
        self.view_encoders_specific = nn.ModuleDict(self.view_encoders_specific)
        self.view_encoders_shared = nn.ModuleDict(self.view_encoders_shared)
        
        #for fuse shared and specific
        emb_dim_sha = self.view_encoders_specific[v_name].get_output_size()
        self.projection_layer = nn.Linear(emb_dim_sha*2, emb_dim_sha)
        
        #for modality prediction
        self.modality_head = nn.LazyLinear(len(self.view_names))
        self.criteria_modality = nn.CrossEntropyLoss()

        #for alignment
        self.criteria_align = nn.L1Loss(reduction='mean')

    def forward(self, 
                views: Dict[str, torch.Tensor],
            intermediate:bool = True,
            out_norm:str="",
            inference_views: list = [],
            **kwargs):
        self.view_encoders = self.view_encoders_specific
        out_zs_views_spe = self.forward_encoders(views)["views:rep"]
        
        self.view_encoders = self.view_encoders_shared
        out_zs_views_sha = self.forward_encoders(views)["views:rep"]
                
        per_view_rep = {"specific":{}, "shared":{}, "fused":{}}
        missing_views = []
        for v_name in self.view_names:
            if len(inference_views) != 0 and v_name not in inference_views:
                missing_views.append(v_name)
                continue

            per_view_rep["specific"][v_name] = out_zs_views_spe[v_name]
            per_view_rep["shared"][v_name]  = out_zs_views_sha[v_name]
            c = torch.cat([per_view_rep["specific"][v_name], per_view_rep["shared"][v_name]], dim=-1)

            per_view_rep["fused"][v_name] = self.projection_layer(c) + per_view_rep["shared"][v_name] #residual over shared
        
        if len(inference_views) != 0: #reconstruct missing view based on avg
            average_feat = torch.mean(torch.stack(list(per_view_rep["shared"].values()), dim=0), dim=0)
            for v_name in missing_views:
                per_view_rep["fused"][v_name] = average_feat

        #merge function
        views_data = [ per_view_rep["fused"][v] for v in self.view_names] # this ensures that the same views are passed for training
        out_z_e = self.fusion_module(views_data, views_available=torch.ones(self.N_views).bool())

        out_y = self.prediction_head(out_z_e["joint_rep"])

        return_dic = {"prediction": self.apply_norm_out(out_y, out_norm)}
        if intermediate:
            return_dic["views:rep"] = per_view_rep
        return return_dic
    
    def loss_batch(self, batch: dict):
        loss_dic = { "predictive": super(ShaSpec, self).loss_batch(batch)["objective"] } #prediction with missing data
        loss_main = self.weights_loss.get("predictive", 1)*loss_dic["predictive"]

        views_dict, _ = self.prepare_batch(batch)
        out_dic = self(views_dict, intermediate=True)
        N_views = len(out_dic["views:rep"]["fused"])
        for v in out_dic["views:rep"]["fused"]:
            if self.weights_loss.get("modality", 0) != 0: #loss is 'activated'
                label_ = np.where(v == np.asarray(self.view_names))[0].item()
                modality_labels = torch.ones_like(out_dic["views:rep"]["specific"][v][:,0], dtype=torch.long)*label_
                loss_dic["Lmod"+v] = self.criteria_modality(self.modality_head(out_dic["views:rep"]["specific"][v]), modality_labels)
                
                loss_main += self.weights_loss["modality"]*loss_dic["Lmod"+v]/N_views

            if self.weights_loss.get("alignment", 0) != 0: #loss is 'activated'
                aux_val = 0
                for v2 in out_dic["views:rep"]["fused"]:
                    if v2 == v:
                        continue
                    aux_val += self.criteria_align(out_dic["views:rep"]["shared"][v], out_dic["views:rep"]["shared"][v2])
                loss_dic["Lalign"+v] = aux_val/(N_views-1)

                loss_main += self.weights_loss["alignment"]*loss_dic["Lalign"+v]/N_views

        return {"objective": loss_main, **loss_dic}
    

class DModDrop(MVFusionMissing):
    #decision-level fusion model with co-learning at feature and decision level
    #missing views are ignored
    def __init__(self,
                 view_models: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 weights_loss: dict = {},
                 **kwargs,
                 ):
        super(DModDrop, self).__init__(view_models, fusion_module, nn.Identity(),
            loss_args=loss_args, view_names=view_names, **kwargs)  
        self.weights_loss = weights_loss
        
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.mutual_distillation = mutual_distillation_loss(self.criteria, temp= self.weights_loss.get("cross_temp", 1))
    
    def loss_batch(self, batch: dict):
        views_dict, target_all = self.prepare_batch(batch)
        out_dic_full = self(views_dict) #full-data prediction only once
        out_dic_full["views:prediction"] = out_dic_full.pop("views:rep") #remove rep to avoid confusion

        ## MISSING DATA PART
        if self.missing_as_aug: #MAUG at decision -level (only play with prediction vectors)
            if self.missing_random: #a single random augmentation instance -- remove in published version, only CoM
                self.all_missing_views = augment_random_missing(self.view_names, perc=self.random_perc)

            views_targets = []
            for i, inference_views in enumerate(self.all_missing_views): 
                views_available_ohv = torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
                perview_pred_masked = {v: out_dic_full["views:prediction"][v] for v in inference_views}

                perview_pred_forced = [perview_pred_masked[v] if v in perview_pred_masked else torch.zeros_like(list(perview_pred_masked.values())[0]) for v in self.view_names]
                out_dic_missing_ = {'prediction': self.fusion_module(perview_pred_forced, views_available=views_available_ohv.bool())["joint_rep"] }                
                if i == 0:
                    out_dic_missing = object_to_list(out_dic_missing_) #to start append values
                else:
                    collate_all_list(out_dic_missing, out_dic_missing_) #add to list in cpu
                views_targets.append(target_all)
            out_dic_missing = stack_all(out_dic_missing, data_type="torch")
            views_targets = torch.concat(views_targets, axis = 0)
            
        ## LOSS PART
        fused_pred = out_dic_full["prediction"]
        pred_view = out_dic_full["views:prediction"]
        N_views = len(self.view_names) 
        loss_dic = {}
        loss_miss = 0
        loss_cons = 0
        loss_mutualind = 0
        loss_mutualcross = 0

        #fused loss
        loss_dic["Lf"] =  self.criteria(fused_pred, target_all) 
        loss_full = self.weights_loss.get("main_pred",1)*loss_dic["Lf"]
        if self.missing_as_aug:
            missing_pred = out_dic_missing["prediction"] #prediction with missing data
            loss_dic["Lm"] =  self.criteria(missing_pred, views_targets) 
            loss_miss += self.weights_loss.get("missing_pred",1)*loss_dic["Lm"]
            if self.weights_loss.get("cml_loss",0) != 0:
                #indenpently if the class is good
                loss_dic["Lcml"] = nn.functional.relu(missing_pred.max(axis=1).values - fused_pred.max(axis=1).values).mean()
                loss_cons += self.weights_loss.get("cml_loss",1)*loss_dic["Lcml"]
            elif self.weights_loss.get("mofe_loss",0) != 0:
                loss_dic["Lmofe"] = nn.functional.relu(loss_dic["Lf"] - loss_dic["Lm"]).mean()
                loss_cons += self.weights_loss.get("mofe_loss",1)*loss_dic["Lmofe"]

        if self.weights_loss.get("mutual_cross", 0) != 0 or self.weights_loss.get("mutual_missing", 0) != 0:
            fused_pred_detach = nn.functional.log_softmax(fused_pred / self.weights_loss.get("cross_temp", 1), dim=-1).detach()
            
        for v_i, v_name in enumerate(self.view_names): #### Decision-level co-learning ####
            if self.weights_loss.get("mutual_ind",0) != 0:
                loss_dic["L"+v_name] = self.criteria(pred_view[v_name], target_all)
                loss_mutualind += self.weights_loss["mutual_ind"]*loss_dic["L"+v_name]/N_views  #maybe another name

            if self.weights_loss.get("mutual_cross", 0) != 0: #self supervised individual data source predictions on full data
                loss_dic["Lmc"+v_name] = self.mutual_distillation(pred_view[v_name], fused_pred_detach)
                loss_mutualcross += self.weights_loss["mutual_cross"]*loss_dic["Lmc"+v_name]/N_views

        total_loss = loss_full +loss_miss + loss_cons + loss_mutualind + loss_mutualcross
        return {"objective": total_loss, **loss_dic}

    def get_encoder_models(self):
        return self.view_encoders
    
    def get_perview_model(self, view_name):
        return nn.ModuleList([self.view_encoders[view_name], self.view_prediction_heads[view_name]])

class Feat4Miss(MVFusionMissing):
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 weights_loss: dict = {},
                 translator_args: dict = {},
                 split_features: bool = False, #whether to split the features in informative and invariant
                 **kwargs,
                 ):
        super(Feat4Miss, self).__init__(view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names, **kwargs)  
        self.weights_loss = weights_loss
        self.missing_as_aug = False
        self.split_features = split_features

        translators = {}
        if self.split_features:
            for v_name ,model_v in view_encoders.items():
                for v_name2 , _ in view_encoders.items():
                    if v_name == v_name2:
                        continue
                    emb_dim = model_v.get_output_size()
                    translators[v_name+"x"+v_name2] = create_model(emb_dim, emb_dim, encoder=True, **translator_args)
        self.translator = nn.ModuleDict(translators) 
        
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**loss_args)
        self.criteria_featdist = nn.MSELoss(reduction='mean')
        self.criteria_contrastive = pairwise_contrastive_loss(self.weights_loss.get("contrastive_temp", 0.1), loss_type="infonce")
        self.criteria_contrastive_sup = supervised_contrastive_loss(self.weights_loss.get("contrastive_temp", 0.1))

    def forward(self, 
                views: Dict[str, torch.Tensor], 
                out_norm:str = "", 
                intermediate:bool=True, 
                inference_views: list=[],
                missing_method: dict = {},
                forward_from_representation: bool = False,
                **kwargs):
        if forward_from_representation: #missing data already applied at representation level
            out_zs_views = views
        else:
            out_zs_views = self.forward_encoders(views, inference_views=inference_views,missing_method=missing_method)["views:rep"]

        aux_reps = {"informative":{}, "invariant":{}}
        if self.split_features:
            for v_name, s_full in out_zs_views.items():
                nfeat = s_full.shape[1] // 2
                aux_reps["informative"][v_name] = s_full[:, :nfeat]
                aux_reps["invariant"][v_name]   =  s_full[:, nfeat:]

        if self.weights_loss.get("cross_translation",0) != 0 and len(inference_views) < len(self.view_names) and len(inference_views) != 0: #reconstruct
            for v_targ in self.view_names:
                if v_targ not in inference_views: #fill out missing view
                    for v_inp in inference_views: #average from all perspective of reconstruction
                        if v_targ in out_zs_views:
                            out_zs_views[v_targ] += self.translator[v_inp+"x"+v_targ](out_zs_views[v_inp])/len(inference_views)
                        else:
                            out_zs_views[v_targ] = self.translator[v_inp+"x"+v_targ](out_zs_views[v_inp])/len(inference_views)
            views_data = [ out_zs_views[v] for v in self.view_names] # this ensures that the same views are passed for training
            views_available_ohv = torch.ones(self.N_views)
       
        else:
            if len(inference_views) != 0 and missing_method.get("name") == "ignore":  #this is usually used with MAUG technique
                views_data = [ out_zs_views[v] for v in self.view_names if v in inference_views] 
            else: #adapt, impute or others
                views_data = [ out_zs_views[v] for v in self.view_names] # this ensures that the same views are passed for training
            views_available_ohv = torch.ones(self.N_views) if len(inference_views) == 0 else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        
        out_z_e = self.fusion_module(views_data, views_available=views_available_ohv.bool())
        
        #prediction head
        out_y = self.prediction_head(out_z_e["joint_rep"])
        return_dic = {"prediction": self.apply_norm_out(out_y, out_norm)}
        if intermediate:
            return_dic["views:rep"] = aux_reps
            return_dic["joint_rep"] = out_z_e["joint_rep"]
        return return_dic

        
    def loss_batch(self, batch: dict):
        views_dict, target_all = self.prepare_batch(batch)
        out_dic_full = self(views_dict) #full-data prediction only once

        if self.missing_as_aug:
            if self.missing_random: #a single random augmentation instance -- remove in published version, only CoM
                missing_case = augment_random_missing(self.view_names, perc=self.random_perc) #return a list of augmentations
                out_dic_missing = self(views_dict, inference_views=missing_case[0], missing_method=self.missing_method)
                views_targets = target_all
            else:
                zs_views = self.forward_encoders(views_dict)["views:rep"]

                views_targets = []
                for i, inference_views in enumerate(self.all_missing_views):
                    if self.missing_method.get("name") == "ignore":
                        aux_zs_views = {k: v for k,v in zs_views.items()} #to force copy dictionary
                    else:
                        value_fill = torch.nan if self.missing_method["value_fill"] == "nan" else self.missing_method["value_fill"] 
                        aux_zs_views = {k: (v if k in inference_views else torch.ones_like(v)*value_fill) for k, v in zs_views.items()}
                        
                    out_dic_ = self(aux_zs_views, inference_views=inference_views, missing_method=self.missing_method, forward_from_representation = True)
                    if i == 0:
                        out_dic_missing = object_to_list(out_dic_) #to start append values
                    else:
                        collate_all_list(out_dic_missing, out_dic_) #add to list in cpu
                    views_targets.append(target_all)
                out_dic_missing = stack_all(out_dic_missing, data_type="torch")
                views_targets = torch.concat(views_targets, axis = 0)

        ## LOSS PART
        fused_pred = out_dic_full["prediction"]
        rep_view = out_dic_full["views:rep"]
        N_views = len(self.view_names) 
        n_combinations = N_views* (N_views-1) #math.factorial(N_views)/(math.factorial(N_views-2))

        loss_dic = {}
        loss_miss = 0
        loss_contr = 0
        loss_disent = 0
        loss_trans = 0
        loss_missdistfeat = 0
        loss_contrgeom = 0

        #fused loss
        loss_dic["Lf"] =  self.criteria(fused_pred, target_all) 
        loss_full = self.weights_loss.get("main_pred",1)*loss_dic["Lf"]
        if self.missing_as_aug:
            missing_pred = out_dic_missing["prediction"] #prediction with missing data
            loss_dic["Lm"] =  self.criteria(missing_pred, views_targets) 
            loss_miss += self.weights_loss.get("missing_pred",1)*loss_dic["Lm"]
            
            if self.weights_loss.get("missing_distillation_feat",0) != 0:
                loss_dic["Lmdf"] = self.criteria_featdist(out_dic_missing["joint_rep"], out_dic_full["joint_rep"].detach())
                loss_missdistfeat += self.weights_loss.get("missing_distillation_feat",1)*loss_dic["Lmdf"]

            if self.weights_loss.get("contrastive_geometric",0) != 0:
                loss_dic["Lcg"] = self.criteria_contrastive(out_dic_full["joint_rep"], out_dic_missing["joint_rep"])
                loss_contrgeom += self.weights_loss.get("contrastive_geometric",1)*loss_dic["Lcg"]

        for v_i, v_name in enumerate(self.view_names):
            if self.weights_loss.get("own_disentangle",0) != 0:
                loss_dic["Ldis"+v_name] = -self.criteria_featdist(rep_view["informative"][v_name], rep_view["invariant"][v_name])
                loss_disent += self.weights_loss["own_disentangle"]*loss_dic["Ldis"+v_name]/N_views
                
            if self.weights_loss.get("cross_translation",0) != 0:
                input_concat_feats = torch.cat([rep_view["informative"][self.view_names[v_i]], rep_view["invariant"][self.view_names[v_i]]], dim=-1)
                
            for v_j in range(N_views): 
                if v_i == v_j:
                    continue
                
                if self.weights_loss.get("contrastive_sup",0) != 0:
                    loss_dic["Lcon"+self.view_names[v_i]+"-"+self.view_names[v_j]] = self.criteria_contrastive_sup(rep_view["invariant"][self.view_names[v_i]], rep_view["invariant"][self.view_names[v_j]],target_all,target_all)
                    loss_contr += self.weights_loss.get("contrastive_sup",0)*loss_dic["Lcon"+self.view_names[v_i]+"-"+self.view_names[v_j]]/n_combinations
                    
                if self.weights_loss.get("cross_translation",0) != 0:
                    target_concat_feats = torch.cat([rep_view["informative"][self.view_names[v_j]], rep_view["invariant"][self.view_names[v_j]]], dim=-1)
                    translated_feat = self.translator[self.view_names[v_i]+"x"+self.view_names[v_j]](input_concat_feats)
                    loss_dic["Ltrans"+self.view_names[v_i]+"x"+self.view_names[v_j]] = self.criteria_featdist(translated_feat, target_concat_feats)
                    loss_trans += self.weights_loss["cross_translation"]*loss_dic["Ltrans"+self.view_names[v_i]+"x"+self.view_names[v_j]]/n_combinations

        total_loss = loss_full + loss_miss + loss_contr + loss_disent + loss_trans + loss_missdistfeat + loss_contrgeom
        return {"objective": total_loss, **loss_dic}
    

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

        #for modality discriminant prediction
        
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
    