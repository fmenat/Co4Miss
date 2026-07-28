
import numpy as np
from pathlib import Path
from sklearn.utils import class_weight


def assign_our_name(config_file, forward_views= [], perc:float=1, more_info_str=""):
    method_name = "Ours"
    
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
    