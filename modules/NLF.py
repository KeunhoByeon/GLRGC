import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture


class NLF:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()
        self.threshold = 0.9

    def step_threshold(self):
        if self.threshold > 0.5:
            self.threshold -= 0.04

    def __call__(self, network, dataloader):
        file_paths, losses = [], []

        network.eval()
        with torch.no_grad():
            for i, (input_paths, inputs, targets, is_noisy) in enumerate(dataloader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                output_ema = network(inputs, ema=True)

                loss = self.criterion(output_ema, targets)
                losses.append(loss.item())

                file_paths.append(input_paths[0])

        max_loss = max(losses)
        normalized_losses = np.array(losses) / max_loss

        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(normalized_losses.reshape(-1, 1))
        probabilities = gmm.predict_proba(normalized_losses.reshape(-1, 1))

        clean_component_index = np.argmin(gmm.means_)
        clean_probabilities = probabilities[:, clean_component_index]

        noise_indices = np.where(clean_probabilities < self.threshold)[0]
        noise_paths = np.array(file_paths)[noise_indices]

        return noise_paths
