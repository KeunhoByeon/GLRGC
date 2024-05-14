import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data_module import *



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
    def __init__(self, dataset_name, stage="train", input_size: int = None, tag=""):
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
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
            ])
            self.random_crop = RandomDynamicCrop((0.2, 1.0))
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])

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

        image = Image.open(img_path).convert('RGB')

        if self.stage == "train" and self.tag != "NLF":
            image_1, image_2 = self.random_crop(image, mult_output=True)
            x1 = self.transform(image_1)
            x2 = self.transform(image_2)
            return img_path, x1, x2, gt, is_noisy
        else:
            x = self.transform(image)
            return img_path, x, gt, is_noisy

    def __len__(self):
        return len(self.data_pair)
