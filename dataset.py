import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import resize_and_pad_image


class NoisyDataset(Dataset):
    def __init__(self, data_dir, input_size: int = None):
        self.input_size = input_size

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.samples = []

        gt_cnt = {}
        for path, dir, files in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                file_index = filename.strip(ext)
                gt = 0

                is_noisy = False
                self.samples.append((os.path.join(path, filename), gt, is_noisy))

        print("Loaded {} samples ({})".format(len(self.samples), gt_cnt))

    def update_noise_labels(self, predictions, threshold=0.5):
        # 예측이 실제 레이블과 다르면 노이즈로 간주
        mismatches = predictions != self.labels
        self.is_noisy = mismatches.float().mean() > threshold

    def __getitem__(self, index, return_path=False):
        img_path, gt, is_noisy = self.samples[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.input_size is not None:
            img = resize_and_pad_image(img, target_size=(self.input_size, self.input_size), keep_ratio=True, padding=True)

        img = img.astype(np.float32) / 255.
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        if return_path:
            return img_path, img, gt, is_noisy
        return img, gt, is_noisy

    def __len__(self):
        return len(self.samples)
