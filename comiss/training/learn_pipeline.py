import copy

import torch
import pytorch_lightning as pl

from comiss.models.fusion_strategy import CoLearnMiss
from comiss.models.nn_models import create_model
from comiss.models.fusion_module import FusionModuleMissing
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
    return model, trainer