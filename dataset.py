import random

import cv2
import numpy as np
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from torchvision import transforms

from data_module import *


def get_transform(args, mode="test"):
    if mode == "train":
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        return iaa.Sequential([
            iaa.Crop(percent=([0., 0.4], [0., 0.4], [0., 0.4], [0., 0.4])),
            iaa.Resize({"height": args.input_size, "width": args.input_size}, interpolation='nearest'),
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            sometimes(
                iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                )),
            iaa.SomeOf((0, 5), [iaa.OneOf([iaa.GaussianBlur((0, 3.0)),
                                           iaa.AverageBlur(k=(2, 7)),
                                           iaa.MedianBlur(k=(3, 11)), ]),
                                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), ], random_order=True)
        ], random_order=True)
    else:
        return iaa.Resize({"height": args.input_size, "width": args.input_size}, interpolation='nearest')


class RandomDynamicCrop(transforms.RandomCrop):
    def __init__(self, scale_range=(0.2, 1.0)):
        super().__init__(size=(0, 0))  # Initialize with dummy values
        self.scale_range = scale_range

    def __call__(self, img, mult_output=False):
        # Compute a random scale size each time the transform is called
        scale = random.uniform(*self.scale_range)
        self.size = (int(img.height * scale), int(img.width * scale))
        if self.size[0] > img.height or self.size[1] > img.width:
            raise ValueError("Crop size must be smaller than or equal to the original image size")

        if mult_output:
            return super().__call__(img.copy()), super().__call__(img.copy())
        return super().__call__(img)


class NoisyDataset(Dataset):
    def __init__(self, args, dataset_name, stage="train", input_size: int = None, tag=""):
        self.args = args
        self.dataset_name = dataset_name
        self.stage = stage
        self.input_size = input_size
        self.tag = tag

        self.noisy_dict = {}
        if dataset_name == "colon":
            self.data_pair = prepare_colon_dataset(stage=stage)
        elif dataset_name == "colon_test2":
            self.data_pair = prepare_colon_test2_dataset(stage=stage)
        elif dataset_name == "gastric":
            self.data_pair = prepare_gastric_dataset(stage=stage)
        elif dataset_name == "prostate_harvard":
            self.data_pair = prepare_prostate_harvard_dataset(stage=stage)
        elif dataset_name == "prostate_ubc":
            self.data_pair = prepare_prostate_ubc_dataset(stage=stage)
        else:
            raise NotImplementedError
        self.update_noise_labels([])

        if self.stage == "train" and self.tag != "NLF":
            self.transform = get_transform(self.args, mode="train")
        else:
            self.transform = get_transform(self.args, mode="test")

        gts = np.array(self.data_pair)[:, 1].astype(int)
        gt_cnt = [np.sum(gts == gt_i) for gt_i in range(4)]
        print("[Dataset {}] Loaded {} samples ({})".format(self.tag, len(self.data_pair), gt_cnt))

    def update_noise_labels(self, noise_labels, logger=None):
        num_clean, num_noise = 0, 0
        for file_path, gt in self.data_pair:
            if file_path not in noise_labels:
                self.noisy_dict[file_path] = False
                num_clean += 1
            else:
                self.noisy_dict[file_path] = True
                num_noise += 1

        if logger is not None:
            logger.print_and_write_log(
                "[Dataset {}] Num Clean {}, Num Noise {}".format(self.tag, num_clean, num_noise))
        else:
            print("[Dataset {}] Num Clean {}, Num Noise {}".format(self.tag, num_clean, num_noise))

    def __getitem__(self, index):
        img_path, gt = self.data_pair[index]
        gt = int(gt)
        is_noisy = self.noisy_dict[img_path]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.stage == "train" and self.tag != "NLF":
            x1 = self.transform(image=image.copy()).astype(float)
            x2 = self.transform(image=image.copy()).astype(float)
            x1 = torch.FloatTensor(x1).permute(2, 0, 1)
            x2 = torch.FloatTensor(x2).permute(2, 0, 1)
            return img_path, x1, x2, gt, is_noisy
        else:
            x = self.transform(image=image).astype(float)
            x = torch.FloatTensor(x).permute(2, 0, 1)
            return img_path, x, gt, is_noisy

    def __len__(self):
        return len(self.data_pair)
