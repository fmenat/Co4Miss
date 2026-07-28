import torch
import torch.nn as nn
import numpy as np

from .losses import pairwise_contrastive_loss, volume_computation, mutual_distillation_loss

class FeatureCoLoss(nn.Module):
    def __init__(self, shared_loss = "contrastive_vol", specific_loss = "modality_disc",
                 contrastive_temp= 0.07, discriminator=None):
        """
        Args:
            shared_loss (str): The type of loss to use for the shared features. Default is "contrastive_vol".
                Options: [contrastive_vol, contrastive_pairs, l1_pairs, l2_pairs]
            specific_loss (str): The type of loss to use for the specific features. Default is "modality_disc".
            contrastive_temp (float): Temperature parameter for contrastive loss. Default is 0.1.
        """
        super().__init__()
        self.shared_loss = shared_loss
        self.specific_loss = specific_loss
        self.contrastive_temp = contrastive_temp

        if self.shared_loss == "contrastive_pairs":
            self.criteria_contrastive = pairwise_contrastive_loss(self.contrastive_temp, loss_type="infonce")

        if self.specific_loss == "modality_disc":
            self.criteria_modality = nn.CrossEntropyLoss()
            self.discriminator = discriminator
        
    def forward(self, modality_shared_features: list, modality_specific_features: list):
        """
        Compute the feature-based co-learning for a batch of features from different modalities.
        Args:
            modality_shared_features (list of torch.Tensor): A list containing modality-shared features from different modalities.
            modality_specific_features (list of torch.Tensor): A list containing modality-specific features from different modalities.
            
            Each tensor should have the shape (batch_size, feature_dim).
            It assumes the first modality is the anchor modality.
        """
        n_samples = modality_shared_features[0].shape[0]
        n_modalities = len(modality_shared_features)
        loss_dic = {}

        if self.shared_loss == "contrastive_vol":
            volume = volume_computation(*modality_shared_features) / self.contrastive_temp
            targets = torch.linspace(0, n_samples - 1, n_samples, dtype=int).to(modality_shared_features[0].device)
            loss_dic[self.shared_loss] = (nn.functional.cross_entropy(-volume, targets, label_smoothing=0.1)  + nn.functional.cross_entropy(-volume.T, targets, label_smoothing=0.1)) / 2  
    
        elif self.shared_loss in ["l1_pairs", "l2_pairs", "contrastive_pairs"]:
            loss_dic[self.shared_loss] = 0
            n_combinations = len(modality_shared_features) * (len(modality_shared_features) - 1) / 2
            for v_i in range(len(modality_shared_features)):
                for v_j in range(len(modality_shared_features)):
                    if v_i != v_j:
                        if self.shared_loss == "l1_pairs":
                            loss_dic["Pairs"+str(v_i)+"-"+str(v_j)] = nn.functional.l1_loss(modality_shared_features[v_i], modality_shared_features[v_j])
                        elif self.shared_loss == "l2_pairs":
                            loss_dic["Pairs"+str(v_i)+"-"+str(v_j)] = nn.functional.mse_loss(modality_shared_features[v_i], modality_shared_features[v_j])
                        elif self.shared_loss == "contrastive_pairs":
                            loss_dic["Pairs"+str(v_i)+"-"+str(v_j)] = self.criteria_contrastive(modality_shared_features[v_i], modality_shared_features[v_j])
                        loss_dic[self.shared_loss] += loss_dic["Pairs"+str(v_i)+"-"+str(v_j)] 
            loss_dic[self.shared_loss] /= n_combinations


        if self.specific_loss == "modality_disc":
            loss_dic[self.specific_loss] = 0
            for v_i in range(len(modality_shared_features)):
                modality_labels = torch.ones(n_samples, dtype=torch.long, device=modality_specific_features[v_i].device)*v_i

                logits = self.discriminator(modality_specific_features[v_i])
                loss_dic["Disc"+str(v_i)] = self.criteria_modality(logits, modality_labels)
                loss_dic[self.specific_loss] += loss_dic["Disc"+str(v_i)]

            loss_dic[self.specific_loss] /= n_modalities

        return loss_dic


class DecisionCoLoss(nn.Module):
    def __init__(self, distillation_loss = "missing_distillation", distillation_temp=0.5, distillation_weight=1):
        """
        Args:
            shared_loss (str): The type of loss to use for the distillation. Default is "missing_distillation".
                Options: [missing_distillation, missing_distillation]
            distillation_temp (float): Temperature parameter for distillation loss. Default is 1.
            distillation_weight (float): Weight for the distillation loss. Default is 1.
        """
        super().__init__()
        self.distillation_loss = distillation_loss
        self.distillation_temp = distillation_temp
        self.distillation_weight = distillation_weight
        self.criteria_ce = nn.CrossEntropyLoss()
        self.mutual_distillation = mutual_distillation_loss(self.criteria_ce, temp= self.distillation_temp) #only computes KL


    def forward(self, modality_specific_preds: list, fused_pred: torch.Tensor, target: torch.Tensor, missing_pred=None):
        n_modalities = len(modality_specific_preds)
        loss_dic = {}

        fused_pred_detach = nn.functional.log_softmax(fused_pred / self.distillation_temp, dim=-1).detach()

        if self.distillation_loss =="missing_distillation":
            loss_dic[self.distillation_loss] = self.criteria_ce(missing_pred, target) + \
                        self.distillation_weight * self.mutual_distillation(missing_pred, fused_pred_detach)

        elif self.distillation_loss =="mutual_distillation":
            loss_dic[self.distillation_loss] = 0
            for v_i in range(len(modality_specific_preds)):
                loss_dic["Dis"+str(v_i)] = self.criteria_ce(modality_specific_preds[v_i], target) + \
                                self.distillation_weight * self.mutual_distillation(modality_specific_preds[v_i], fused_pred_detach)
                loss_dic[self.distillation_loss] += loss_dic["Dis"+str(v_i)]
            loss_dic[self.distillation_loss] /= n_modalities

        return loss_dic


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    feature_dim = 128
    n_modalities = 3

    modality_shared_features = [torch.randn(batch_size, feature_dim) for _ in range(n_modalities)]
    modality_specific_features = [torch.randn(batch_size, feature_dim) for _ in range(n_modalities)]

    feature_co_loss = FeatureCoLoss(shared_loss="contrastive_vol", specific_loss="modality_disc")
    loss_dict = feature_co_loss(modality_shared_features, modality_specific_features)
    print(">>>> contrastive_vol: ", loss_dict)

    feature_co_loss = FeatureCoLoss(shared_loss="contrastive_pairs", specific_loss="modality_disc")
    loss_dict = feature_co_loss(modality_shared_features, modality_specific_features)
    print(">>>> contrastive_pairs: ", loss_dict)

    feature_co_loss = FeatureCoLoss(shared_loss="l2_pairs", specific_loss="modality_disc")
    loss_dict = feature_co_loss(modality_shared_features, modality_specific_features)
    print(">>>> l2_pairs: ", loss_dict)

    modality_specific_preds = [torch.randn(batch_size, 10) for _ in range(n_modalities)]
    fused_pred = torch.stack(modality_specific_preds).mean(dim=0)
    target = torch.randint(0, 10, (batch_size,))

    decision_co_loss = DecisionCoLoss(distillation_loss="mutual_distillation", distillation_temp=0.5, distillation_weight=1)
    loss_dict_decision = decision_co_loss(modality_specific_preds, fused_pred, target)
    print(">>>> decision mutual distillation: ", loss_dict_decision)

    random_selected_modality = np.random.choice(n_modalities, size=2) #ModDrop-style
    missing_pred = torch.stack([modality_specific_preds[i] for i in random_selected_modality]).mean(dim=0)
    decision_co_loss = DecisionCoLoss(distillation_loss="missing_distillation", distillation_temp=0.5, distillation_weight=1)
    loss_dict_decision = decision_co_loss(modality_specific_preds, fused_pred, target, missing_pred)
    print(">>>> decision missing distillation: ", loss_dict_decision)

    