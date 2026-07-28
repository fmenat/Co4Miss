import torch
from torch import nn

from pytorch_metric_learning import losses


def get_loss_by_name(name, **loss_args):
    #https://pytorch.org/docs/stable/nn.html#loss-functions
    name = name.strip().lower().replace("_","")
    if "n_labels" in loss_args:
        loss_args = dict(loss_args)
        loss_args.pop("n_labels")
        
    if ("cross" in name and "entr" in name) or name=="ce":
        return nn.CrossEntropyLoss(reduction="mean", **loss_args)
    elif ("bin" in name and "entr" in name) or name=="bce":
        return nn.BCEWithLogitsLoss(reduction="mean", **loss_args)
    elif name == "kl" or name=="divergence": 
        return nn.KLDivLoss(reduction="mean")
    elif name == "mse" or name =="l2":
        return nn.MSELoss(reduction='mean')
    elif name == "mae" or name =="l1":
        return nn.L1Loss(reduction='mean')
    elif name =="norm-mse":
        return normalize_mse(**loss_args)


def normalize_mse(mean, std, **kwargs):
    def real_loss(pred, real):
        return nn.MSELoss(reduction='mean')((pred-mean)/std, (real-mean)/std)
    return real_loss

def contrastive_loss(tau=0.1, normalize=True):
    def real_loss(x_q,x_k):
        if normalize:
            x_q = nn.functional.normalize(x_q, dim=1)
            x_k = nn.functional.normalize(x_k, dim=1)
        sim_matrix = torch.mm(x_q, x_k.t()) #n_samples x n_samples
        identity_ = torch.arange(sim_matrix.shape[0]).to(sim_matrix.device).long()
        return nn.CrossEntropyLoss()(sim_matrix/tau, identity_)
    return real_loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cross_bt_loss(dim_common, norm_func, lambda_=1):
    def real_loss(x_q, x_k):
        x_q = norm_func(x_q)
        x_k = norm_func(x_k)
        cross_matrix = torch.mm(x_q.t(), x_k) #n_dims x n_dims
        c_c = cross_matrix[:dim_common,:dim_common]
        c_u = cross_matrix[dim_common:,dim_common:]

        on_diag_c = (torch.diagonal(c_c) -1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()

        return {"common": on_diag_c + lambda_ * off_diag_c, "unique": on_diag_u + lambda_ * off_diag_u}
    return real_loss

def single_bt_loss(norm_func, lambda_=1):
    def real_loss(x_q, x_k):
        # empirical cross-correlation matrix
        c = norm_func(x_q).T @ norm_func(x_k)
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambda_ * off_diag
        return {"loss":loss, "on_diag": on_diag, "off_diag":off_diag}
    return real_loss

def pairwise_contrastive_loss(temperature=0.1, loss_type="infonce"):
    if loss_type == "infonce":
        loss_fn = losses.NTXentLoss(temperature=temperature)
    meta_loss = losses.SelfSupervisedLoss(loss_fn, symmetric=True)

    def real_loss(x_q, x_v):
        return meta_loss(embeddings=x_q, ref_emb=x_v)
    return real_loss

def supervised_contrastive_loss(temperature=0.1):
    loss_fn = losses.SupConLoss(temperature=temperature)

    def real_loss(x_q, x_v, labels_q, labels_v):
        return loss_fn(embeddings=x_q, ref_emb=x_v, labels=labels_q, ref_labels=labels_v)
    return real_loss


def mutual_distillation_loss(criteria, temp: float = 1.0):

    def real_loss(pred_v,true_v):
        if type(criteria) == nn.BCEWithLogitsLoss:
            return (temp**2)*nn.functional.binary_cross_entropy_with_logits(pred_v/temp, true_v)
        else:
            pred_view_v_logsoftmax = nn.functional.log_softmax(pred_v/temp, dim=-1)
            return (temp**2)*nn.functional.kl_div(pred_view_v_logsoftmax, true_v, log_target=True, reduction="batchmean")
    return real_loss


#from https://github.com/ispamm/GRAM
def volume_computation_old(anchor, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]
    inputs_s = torch.stack(inputs, dim=0)      # (N, B, D)
    inputs_s = nn.functional.normalize(inputs_s, dim=-1) #normalize
    anchor = nn.functional.normalize(anchor, dim=-1) #normalize

    # Compute pairwise dot products for anchor with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2) #(B, B)
    # Compute pairwise dot products for anchor with each input
    l_inputs = [anchor @ inp.T for inp in inputs_s]
    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs_s):
        row = []
        for j, input2 in enumerate(inputs_s):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)
    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs_s))]
    ], dim=-2)
    
    G = G.contiguous()
    # Compute the determinant for each Gram matrix
    
    #ORIGINAL VERSION (SLOW AND INESTABLE)
    #gram_det = torch.det(G.float())
    # Compute the square root of the absolute value of the determinants
    #return torch.sqrt(torch.abs(gram_det))

    #LOG-WAY
    if len(inputs) >= 10:
        eigvals = torch.linalg.eigvalsh(G)  # (B, B, N+1)
        eps = 1e-6
        eigvals = torch.clamp(eigvals, min=eps)# clamp negative & tiny eigenvalues
        logabsdet = eigvals.log().sum(dim=-1)
    else:
        sign, logabsdet = torch.linalg.slogdet(G)

    return torch.exp(0.5 * logabsdet)
    

def volume_computation(anchor, *inputs):
    batch_size2 = inputs[0].shape[0]
    inputs_s = torch.stack(inputs, dim=0)      # (N, B, D)
    inputs_s = nn.functional.normalize(inputs_s, dim=-1) #normalize
    anchor = nn.functional.normalize(anchor, dim=-1) #normalize

     # Compute pairwise dot products for anchor with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2) #(B, B)

    # Compute pairwise dot products for anchor with each input
    l_inputs = torch.matmul(anchor, inputs_s.transpose(-1, -2)) # result: (N, B, B)

    # dot products over D
    input_dot_products = torch.einsum('nbd,mbd->nmb', inputs_s, inputs_s) # result: (N, N, B)
    input_dot_products = input_dot_products.unsqueeze(-1).expand(-1, -1, -1, batch_size2)  # (N, N, B, B)

    top = torch.cat( [aa.unsqueeze(0), l_inputs], dim=0)  # (N+1, B, B)
    rest = torch.cat(
        [
            l_inputs.unsqueeze(1),   # (N, 1, B, B)
            input_dot_products # (N, N, B, B)
        ],
        dim=1
    )  # (N, N+1, B, B)
    G = torch.cat( [top.unsqueeze(0), rest],  dim=0)  # (N+1, N+1, B, B)
    G = G.permute(2, 3, 0, 1)  # (B, B, N+1, N+1)

    G = G.contiguous()

    #EFFICIENT WAY VIA CHOLESKY (UNSTABLE -- SOMETIMES FAILS)
    #L = torch.linalg.cholesky(G)   # (B, B, N+1, N+1)
    #logabsdet = 2.0 * torch.sum( torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)

    #LOG-WAY
    if len(inputs) >= 10:
        eigvals = torch.linalg.eigvalsh(G)  # (B, B, N+1)
        eps = 1e-6
        eigvals = torch.clamp(eigvals, min=eps)# clamp negative & tiny eigenvalues
        logabsdet = eigvals.log().sum(dim=-1)
    else:
        sign, logabsdet = torch.linalg.slogdet(G)

    return torch.exp(0.5 * logabsdet)
    
    