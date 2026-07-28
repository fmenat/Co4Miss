import numpy as np
import itertools
import torch

def possible_missing_mask(n_views, skip_full=False):
    #all possible of combination/permutation for use of views with "n_views" number of views
    return (2**n_views)- (2 if skip_full else 1) #or len(create_list_options(n_views))

def create_list_options(n_views, skip_full=False):
    #create a list with all the combination/permutation of the views. To be extracted on each index i
    return list(itertools.product([0, 1], repeat=n_views))[1:(-1 if skip_full else None)][::-1]


def augment_all_missing(view_names_forward, skip_full=False):
    #return a list of all the view names for augmentation ["S2", ["S2", "weather"], "weather", ...]
    n_views = len(view_names_forward)
    
    n_missing_mask = possible_missing_mask(n_views, skip_full=skip_full)
    lst_missing = create_list_options(n_views, skip_full=skip_full)
    
    if n_views > 10: #forcing max 1024 options to avoid memory issues
        idx_r = np.random.choice(n_missing_mask, size=2**10, replace=False)
        lst_missing = np.asarray(lst_missing)[idx_r].tolist()
        n_missing_mask = len(lst_missing)
    
    augmented_view_list = []
    for i in range(n_missing_mask):
        views_mask_i = np.asarray(lst_missing[i]).astype(bool) #mask as [0,1,1]

        augmented_view_list.append(np.asarray(view_names_forward)[views_mask_i].tolist())    
    return augmented_view_list

def augment_random_missing(view_names_forward, perc= 0, skip_full=False):
    #return a single augmentation based on missing
    n_views = len(view_names_forward)
    
    if perc != 0 and perc != 1:
        affected_views = []
        while (len(affected_views) == 0):
            for v in view_names_forward:
                if np.random.rand() > perc:
                    affected_views.append(v)

    else: #1/combinations randomness
        n_missing_mask = possible_missing_mask(n_views, skip_full=skip_full )
        lst_missing = create_list_options(n_views, skip_full=skip_full)
        if skip_full:
            n_missing_mask -= 1
            lst_missing = lst_missing[1:] #remove full data option
        
        i_rnd = np.random.randint(0, n_missing_mask)
        views_mask_i = np.asarray(lst_missing[i_rnd]).astype(bool) #mask as [0,1,1]
        affected_views =  np.asarray(view_names_forward)[views_mask_i].tolist()
        
    return [affected_views]

def augment_macro(view_names, rand: bool = False, perc = 0, skip_full=False ):
    if rand:
        return augment_random_missing(view_names, perc=perc, skip_full=skip_full)
    else:
        return augment_all_missing(view_names, skip_full=skip_full)
    
def create_awareness_vectors(dim, view_names, init="zeros"):
    if init =="zeros":
        return {v: torch.nn.Parameter(torch.zeros(dim)) for v in view_names}
    elif init == "normal":
        return {v: torch.nn.Parameter(torch.normal(0,1, size=(dim,))) for v in view_names}
    elif init == "random":
        return {v: torch.nn.Parameter(torch.rand(dim)) for v in view_names}
    elif init == "uniform":
        return {v: torch.nn.Parameter(torch.FloatTensor(dim).uniform_(-1, 1)) for v in view_names}

if __name__ == "__main__":
    view_names = [ "S1", "S2","weather"]
    skip_full = True
    
    print("View names =",view_names)
    print("total options for augmentation", possible_missing_mask(len(view_names), skip_full=skip_full))
    print("Augmented views = ",augment_all_missing(view_names, skip_full=skip_full))
    print("Random augmented views = ",augment_random_missing(view_names, skip_full=skip_full))

    print("Random with perc 10 of augmented views = ",augment_random_missing(view_names, perc=0.10, skip_full=skip_full  ))
    print("Random with perc 30 of augmented views = ",augment_random_missing(view_names, perc=0.30, skip_full=skip_full))
    print("Random with perc 50 of augmented views = ",augment_random_missing(view_names, perc=0.50, skip_full=skip_full))
    print("Random with perc 70 of augmented views = ",augment_random_missing(view_names, perc=0.70, skip_full=skip_full))
    print("Random with perc 90 of augmented views = ",augment_random_missing(view_names, perc=0.90, skip_full=skip_full))