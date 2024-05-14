import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections):
        """
        projections: 배치 내 모든 샘플의 투영된 특징 벡터 (2N, 128)
        """
        if len(projections) == 0:
            return torch.tensor(0., device=projections.device)

        labels = torch.arange(len(projections) // 2)
        labels = torch.cat((labels, labels)).to(projections.device)

        norms = projections.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(projections, projections.t()) / (norms * norms.t())

        labels = labels.unsqueeze(0)
        mask = torch.eq(labels, labels.t()).float()

        positive_mask = mask.fill_diagonal_(0)

        exp_similarities = torch.exp(similarity_matrix / self.temperature)
        exp_similarities = exp_similarities * positive_mask  # positive pair만을 고려
        sum_exp_similarities = exp_similarities.sum(dim=1, keepdim=True)

        log_prob = similarity_matrix - torch.log(sum_exp_similarities + 1e-8)
        loss = -torch.sum(log_prob * positive_mask) / positive_mask.sum()

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
