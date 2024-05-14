import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(LocalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections):
        if projections.size(0) == 0:
            return torch.tensor(0., device=projections.device)

        # 레이블 생성
        labels = torch.arange(projections.size(0) // 2, device=projections.device)
        labels = torch.cat((labels, labels))

        # 코사인 유사도 계산
        norms = projections.norm(dim=1, keepdim=True)
        similarity_matrix = torch.mm(projections, projections.t()) / (norms * norms.t())

        # 자기 자신과의 유사도를 -inf로 설정
        similarity_matrix.fill_diagonal_(float('-inf'))

        # 유사도 행렬을 소프트맥스 적용 전에 exp 취하기
        exp_similarities = torch.exp(similarity_matrix / self.temperature)

        # Positive 마스크 생성
        labels = labels.unsqueeze(0)
        positive_mask = torch.eq(labels, labels.t()).float()
        positive_mask.fill_diagonal_(0)

        # 소프트맥스 분모 계산
        sum_exp_similarities = exp_similarities.sum(dim=1, keepdim=True)

        # 로그 확률 계산
        log_prob = torch.log(exp_similarities / sum_exp_similarities)
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
