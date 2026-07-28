
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight

def assign_multifusion_name(training = {}, method = {}, forward_views= [], perc:float=1,  more_info_str = ""):
    method_name = ""
    if method.get("colearn"):
        method_name += f"Co"
    if method["feature"]:
        method_name += f"Feat_{method['agg_args']['mode']}"
    else:
        method_name += f"Dec_{method['agg_args']['mode']}"

    if "adaptive" in method["agg_args"]:
        if method["agg_args"]["adaptive"]:
            method_name += "_GF"
    if "features" in method["agg_args"]:
        if method["agg_args"]["features"]:
            method_name += "f"

    if "multi" in training["loss_args"]:
        if training["loss_args"]["multi"]:
            method_name += "_MuLoss"

    if training.get("missing_as_aug"):
        if training["missing_method"].get("missing_random"):
            method_name += "-SD"
        else:
            method_name += "-MAUG"
    if training.get("missing_method"):
        method_name += f"-{training['missing_method']['name']}" + (f"_{training['missing_method']['where']}" if training['missing_method'].get("where") else "")
    
    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"

    return method_name + more_info_str

def assign_our_name(config_file, forward_views= [], perc:float=1, more_info_str=""):
    method_name = "Ours"
    
    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"

    return method_name + more_info_str

def assign_pool_name(config_file, forward_views= [], perc:float=1, more_info_str=""):
    method_name = "PoolSeq" if config_file.get("sequential_flag") else "Pool"

    if config_file["architecture"]["predictive_model"].get("sharing"):
        method_name += "SensInv"

    if config_file["training"].get("additional_args"):
        if config_file["training"]["additional_args"].get("cotraining_flag"):
            method_name += "CoTrain"

    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"

    return method_name + more_info_str

def assign_labels_weights(config_file, data_):
    train_data_target = data_.get_all_labels().astype(int).flatten()
    if type(data_.target_names) == list:
        config_file["training"]["loss_args"]["n_labels"] = len(data_.target_names)  #train_data_target.max() +1
    else:
        config_file["training"]["loss_args"]["n_labels"] = train_data_target.max() +1
    config_file["training"]["loss_args"]["weight"] = class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
    