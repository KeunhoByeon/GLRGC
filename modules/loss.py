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
    def __init__(self, temperature=0.05):
        super(GlobalRelationLoss, self).__init__()
        self.temperature = temperature  # 0.05?
        self.epsilon = 1e-8

    def forward(self, features_s, features_t):
        similarities_s = torch.mm(features_s, features_s.t()) / self.temperature
        similarities_t = torch.mm(features_t, features_t.t()) / self.temperature

        p_s = F.softmax(similarities_s, dim=1) + self.epsilon
        p_t = F.softmax(similarities_t, dim=1) + self.epsilon

        kl_st = F.kl_div(p_s.log(), p_t, reduction='batchmean')
        kl_ts = F.kl_div(p_t.log(), p_s, reduction='batchmean')

        loss = 0.5 * (kl_st + kl_ts)

        return loss
