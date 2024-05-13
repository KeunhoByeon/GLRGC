import torch
import torch.nn.functional as F


def local_contrastive_loss(embeddings, labels, temperature=0.5):
    device = embeddings.device
    batch_size = embeddings.size(0)

    # 유사도 행렬 계산
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2) / temperature

    # 레이블 기반 마스크 생성
    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # 손실 계산
    sim_matrix = sim_matrix - torch.eye(batch_size).to(device) * 1e12  # 자기 자신과의 유사도 제거
    exp_sim = torch.exp(sim_matrix)
    pos_sum = (exp_sim * mask).sum(dim=1)  # 긍정적 쌍의 합
    all_sum = exp_sim.sum(dim=1)  # 모든 쌍의 합

    loss = -torch.log(pos_sum / all_sum).mean()  # 대조적 손실

    return loss


def global_relation_loss(features_s, features_t, temperature=0.05):
    similarities_s = torch.mm(features_s, features_s.t()) / temperature
    similarities_t = torch.mm(features_t, features_t.t()) / temperature

    p_s = F.softmax(similarities_s, dim=1)
    p_t = F.softmax(similarities_t, dim=1)

    kl_st = F.kl_div(p_s.log(), p_t, reduction='batchmean')
    kl_ts = F.kl_div(p_t.log(), p_s, reduction='batchmean')

    loss = 0.5 * (kl_st + kl_ts)

    return loss


def consistency_loss(output1, output2, weight=1.0):
    return weight * F.mse_loss(output1, output2)
