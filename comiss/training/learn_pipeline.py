import copy

import torch
import pytorch_lightning as pl

from comiss.models.fusion_strategy import Feature4Missing, FeatureFusion, DecisionFusion, SingleViewPool,ShaSpecMV, DModDropMV, Feature4Missing
from comiss.models.fusion_strategy import CoLearnMiss
from comiss.models.nn_models import create_model
from comiss.models.fusion_module import FusionModuleMissing
from comiss.models.utils import get_dic_emb_dims
from comiss.training.pipeline_utils import prepare_callback, build_dataloaders, get_shape_view, extract_n_labels


def ours_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, method = {}, architecture={}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    model_view_names =  train_data.used_view_names
    folder_c = output_dir_folder+"/run-saves"
    n_labels = extract_n_labels(loss_args)

    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for view_n in model_view_names:
        actual_shape_v = get_shape_view(view_n, train_data, architecture=architecture)
        if "-ALL" in architecture["encoders"]:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"]["-ALL"])
        else:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"][view_n])
    #MODEL DEFINITION -- Fusion-Part
    args_model = {"loss_args": loss_args, **training.get("additional_args", {})}
    
    method["agg_args"]["emb_dims"] = [n_labels for _ in range(len(views_encoder))]
    fusion_module = FusionModuleMissing(**method["agg_args"])

    pred_base = create_model(emb_dim, None, just_base=True, **architecture["predictive_model"]) 
    model = CoLearnMiss(views_encoder, pred_base, fusion_module, n_labels=n_labels, view_names=list(views_encoder.keys()), head_args=architecture["predictive_model"],**args_model)
    print("Initial parameters of model:", model.hparams_initial)

    if "missing_as_aug" in training:
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))
                
    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2), monitor_metric=training.get("monitor_metric","objective"))
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"], **training.get("trainer_args", {}))
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    if hasattr(model, 'awl'):
        print("Learned weights", model.awl.get_params())
    return model, trainer


def MultiFusion_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, method = {}, architecture={}, pretrained_encoders={},  **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    model_view_names =  train_data.used_view_names
    folder_c = output_dir_folder+"/run-saves"
    n_labels = extract_n_labels(loss_args)

    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for view_n in model_view_names:
        actual_shape_v = get_shape_view(view_n, train_data, architecture=architecture)
            
        if len(pretrained_encoders) != 0 and view_n in pretrained_encoders: 
            views_encoder[view_n] = copy.deepcopy(pretrained_encoders[view_n])
            views_encoder[view_n].load_state_dict(pretrained_encoders[view_n].state_dict())
        elif "-ALL" in architecture["encoders"]:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"]["-ALL"])
        else:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"][view_n])
    #MODEL DEFINITION -- Fusion-Part
    args_model = {"loss_args": loss_args, **training.get("additional_args", {})}
    
    if method["feature"]:
        method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
        fusion_module = FusionModuleMissing(**method["agg_args"])
        input_dim_task_mapp = fusion_module.get_info_dims()["joint_dim"]

        predictive_model = create_model(input_dim_task_mapp, n_labels, **architecture["predictive_model"], encoder=False) 
    
        if method.get("shaspec"):
            shared_view_encoders = create_model(emb_dim, emb_dim, **architecture["shared_encoder"], encoder=True) 
            model = ShaSpecMV(views_encoder, fusion_module, predictive_model, shared_view_encoders, view_names=list(views_encoder.keys()), **args_model)
        elif method.get("simmmdg") or method.get("missmodal"):
            model = Feature4Missing(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), translator_args=architecture.get("translator_args"), **args_model)
        else:
            model = FeatureFusion(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), **args_model)

    else:
        method["agg_args"]["emb_dims"] = [n_labels for _ in range(len(views_encoder))]
        fusion_module = FusionModuleMissing(**method["agg_args"])
        
        pred_base = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False)  
        prediction_models = {}
        for view_n in views_encoder:
            if architecture["predictive_model"].get("sharing"):
                pred_ = pred_base
            else:
                pred_ = copy.deepcopy(pred_base)
                pred_.load_state_dict(pred_base.state_dict())  
            prediction_models[view_n] = torch.nn.Sequential(views_encoder[view_n], pred_)
            prediction_models[view_n].get_output_size = pred_.get_output_size

        if method.get("modrop"):    
            model = DModDropMV(prediction_models, fusion_module=fusion_module,view_names=list(views_encoder.keys()),**args_model)    
        else:
            model = DecisionFusion(view_encoders=prediction_models, fusion_module=fusion_module,view_names=list(views_encoder.keys()),**args_model)    

    if "missing_as_aug" in training:
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))
    
    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"], **training.get("trainer_args", {}))
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    if hasattr(model, 'awl'):
        print("Learned weights", model.awl.get_params())
    return model, trainer


def PoolEnsemble_train(train_data: dict, val_data = None,
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, architecture={}, checkpoints=True, **kwargs):
    folder_c = output_dir_folder+"/run-saves"
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    model_view_names = train_data.used_view_names
    n_labels = extract_n_labels(loss_args)

    args_model =  {"loss_args":loss_args, "view_names":model_view_names, **training.get("additional_args", {})}
       
    #MODEL DEFINITION -- ENCODER
    pred_base = create_model(emb_dim, n_labels, encoder=False, **architecture["predictive_model"]) 
    views_encoder  = {}
    view_prediction_heads = {}
    for view_n in model_view_names:
        actual_shape_v = get_shape_view(view_n, train_data, architecture=architecture)
        if "-ALL" in architecture["encoders"]:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"]["-ALL"])
        else:
            views_encoder[view_n] = create_model(actual_shape_v, emb_dim, **architecture["encoders"][view_n])
        
        if architecture["predictive_model"].get("sharing"):
            view_prediction_heads[view_n] = pred_base
        else:
            view_prediction_heads[view_n] = copy.deepcopy(pred_base)
            view_prediction_heads[view_n].load_state_dict(pred_base.state_dict())   
    model = SingleViewPool(views_encoder, view_prediction_heads, **args_model)
    print("Initial parameters of model:", model.hparams_initial)
    
    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    if checkpoints:
        extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    else:
        extra_objects = {"callbacks": []}
    
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"], **training.get("trainer_args", {}))
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

    if checkpoints:
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    if training.get("inference"):
        model.set_aggregation(training.get("inference")["mode"])

    return model, trainer

def SequentialPoolEnsemble_train(train_data: dict, val_data = None,
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, architecture={}, pretrained_encoders={}, **kwargs):
    folder_c = output_dir_folder+"/run-saves"
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    model_view_names = train_data.used_view_names
    n_labels = extract_n_labels(loss_args)

    #DATA DEFINITION --
    train_dataloader, val_dataloader, monitor_name = build_dataloaders(train_data, val_data, batch_size=batch_size, parallel_processes=training.get("parallel_processes",2))
    
    views_encoder_pool = {}
    view_prediction_heads_pool = {}
    
    for view_n in (model_view_names if len(pretrained_encoders) == 0 else pretrained_encoders.keys()):
        print("------------------------------------------------ Training view:", view_n, "------------------------------------------------")
        args_model =  {"loss_args":loss_args, "view_names":[view_n], **training.get("additional_args", {})}
    
        #MODEL DEFINITION -- ENCODER
        if len(pretrained_encoders) == 0 or view_n not in pretrained_encoders:
            actual_shape_v = get_shape_view(view_n, train_data, architecture=architecture)
            if "-ALL" in architecture["encoders"]:
                view_encoder = create_model(actual_shape_v, training["emb_dim"], **architecture["encoders"]["-ALL"])
            else:
                view_encoder = create_model(actual_shape_v, training["emb_dim"], **architecture["encoders"][view_n])
        else:
            if training.get("freeze_pretrain", False):
                view_encoder = pretrained_encoders[view_n]
                for param in view_encoder.parameters():
                    param.requires_grad = False
            else:
                view_encoder = copy.deepcopy(pretrained_encoders[view_n])
                view_encoder.load_state_dict(pretrained_encoders[view_n].state_dict())
        emb_dim = view_encoder.get_output_size()
            
        # PREDICTION HEAD BASE
        view_prediction_head = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False )
        
        model = SingleViewPool({view_n: view_encoder}, {view_n: view_prediction_head}, **args_model)
        
        #FIT
        extra_objects = prepare_callback(data_name, method_name+f"-{view_n}", run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"], **training.get("trainer_args", {}))
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

        views_encoder_pool[view_n] = model.get_encoder_models()[view_n]
        view_prediction_heads_pool[view_n] = model.get_sv_models()[view_n]

    print("------------------------------------------------ Training finished, now creating the pool of all sub-models ------------------------------------------------")
    full_model = SingleViewPool(views_encoder_pool, view_prediction_heads_pool, **args_model)
    if training.get("inference"):
        full_model.set_aggregation(training.get("inference")["mode"])

    return full_model, trainer