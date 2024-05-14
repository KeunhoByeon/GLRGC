import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=1.0):
        super(LocalContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, embeddings, labels):
        if embeddings.size(0) == 0:
            return torch.tensor(0., device=embeddings.device)

        distance_matrix = self.pdist(embeddings, squared=False)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # 같은 클래스 내의 샘플 거리는 0에 가깝게, 다른 클래스는 margin 이상이 되게 유도
        positive_loss = (1 - mask) * F.relu(self.margin - distance_matrix) ** 2
        negative_loss = mask * distance_matrix ** 2

        loss = positive_loss + negative_loss
        return loss.mean()

    def pdist(self, embeddings, squared=False):
        e_square = embeddings.square().sum(dim=1)
        prod = embeddings @ embeddings.T
        dist = e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod
        dist = dist.clamp(min=0)
        if not squared:
            dist = dist.sqrt()
        return dist


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
