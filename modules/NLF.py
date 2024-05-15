import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
                curr_losses = F.cross_entropy(output_ema, targets, reduction='none')
                losses.extend(curr_losses.tolist())
                file_paths.extend(input_paths.tolist())

        max_loss = max(losses)
        normalized_losses = np.array(losses) / max_loss

        gmm = GaussianMixture(n_components=2).fit(normalized_losses.reshape(-1, 1))
        gmm_labels = gmm.predict_proba(normalized_losses.reshape(-1, 1))

        noisy_samples = [file_paths[i] for i in range(len(file_paths)) if gmm_labels[i, 1] > self.threshold]

        return noisy_samples
