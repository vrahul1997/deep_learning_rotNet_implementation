import albumentations
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data.dataset import Dataset


class RotnetDataset(Dataset):
    def __init__(self, img_paths, targets=None, resize=None):
        self.images = img_paths
        self.targets = targets
        self.resize = resize

        self.aug = albumentations.Compose(
            [albumentations.Normalize(always_apply=True)]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert("RGB")

        if self.resize is not None:
            img.resize((self.resize[1], self.resize[0]),
                       resample=Image.BILINEAR)

        img = np.array(img)
        aug_image = self.aug(image=img)
        image = aug_image["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        if self.targets is not None:
            targets = self.targets[item]

            return {
                "images": torch.tensor(image, dtype=torch.float),
                "target": torch.tensor(targets, dtype=torch.long),
            }

        else:
            return {
                "images": torch.tensor(image, dtype=torch.float32),
            }
