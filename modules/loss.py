import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections):
        """
        projections: The projected feature vectors of all samples in the batch (2N, 128)
        """
        if len(projections) == 0:
            return torch.tensor(0., device=projections.device)

        # Normalize the projections
        projections = F.normalize(projections, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.mm(projections, projections.t())

        # Generate labels
        labels = torch.arange(len(projections) // 2).to(projections.device)
        labels = torch.cat((labels, labels))

        # Create mask to identify positive pairs
        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.t()).float()
        positive_mask = mask.fill_diagonal_(0)

        # Apply temperature
        similarity_matrix /= self.temperature

        # Compute exponentiated similarity scores
        exp_similarities = torch.exp(similarity_matrix)

        # Sum of similarities for denominator
        sum_exp_similarities = exp_similarities.sum(dim=1, keepdim=True) - exp_similarities.diag().view(-1, 1)

        # Compute log probabilities
        log_prob = similarity_matrix - torch.log(sum_exp_similarities + 1e-8)

        # Compute the local contrastive loss
        try:
            loss = -torch.sum(log_prob * positive_mask) / positive_mask.sum()
        except:
            import pdb; pdb.set_trace()

        return loss


class GlobalRelationLoss(nn.Module):
    def __init__(self):
        super(GlobalRelationLoss, self).__init__()

    def calculate_relation_matrix(self, features):
        relation_matrix = torch.matmul(features, features.T)
        relation_matrix = F.normalize(relation_matrix, p=2, dim=1)  # L2 row normalization
        return relation_matrix

    def symmetric_kl_divergence(self, p, q):
        kl_pq = F.kl_div(p.log(), q, reduction='batchmean')
        kl_qp = F.kl_div(q.log(), p, reduction='batchmean')
        return 0.5 * (kl_pq + kl_qp)

    def forward(self, features_s, features_t):
        relation_matrix_s = self.calculate_relation_matrix(features_s)
        relation_matrix_t = self.calculate_relation_matrix(features_t)

        loss = self.symmetric_kl_divergence(relation_matrix_s, relation_matrix_t)
        return loss
