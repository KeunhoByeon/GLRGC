import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 모델, 데이터셋 및 데이터로더 설정 예
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 데이터셋 및 데이터로더 설정
data = torch.randn(100, 10)
labels = torch.randn(100, 1)
dataset = torch.utils.data.TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10)

# 모델 및 옵티마이저 설정
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 손실 계산 및 정규화
losses = []
model.eval()
with torch.no_grad():
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
max_loss = max(losses)
normalized_losses = np.array(losses) / max_loss

# GMM 적합 및 클린 확률 계산
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(normalized_losses.reshape(-1, 1))
probabilities = gmm.predict_proba(normalized_losses.reshape(-1, 1))

# 클린 확률 추출 (클린 컴포넌트가 평균이 더 작은 컴포넌트라고 가정)
clean_component_index = np.argmin(gmm.means_)
clean_probabilities = probabilities[:, clean_component_index]

# 임계값 설정 및 노이즈 샘플 식별
threshold = 0.9  # 초기 임계값
noise_indices = np.where(clean_probabilities < threshold)[0]

# 출력
print("Identified noise indices:", noise_indices)
