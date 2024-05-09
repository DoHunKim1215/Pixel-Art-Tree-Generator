import os

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def train_dataloader(img_path, batch_size=64, num_worker=0):
    dataloader = DataLoader(
        PixelArtDataset(img_path),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_worker,
        pin_memory=True,
        persistent_workers=True
    )

    return dataloader


def test_dataloader(img_path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        PixelArtDataset(img_path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


class PixelArtDataset(Dataset):
    leaf_types = {'acacia': 0, 'bush': 1, 'shrub': 2, 'pine': 3, 'oak': 4, 'palm': 5, 'poplar': 6, 'willow': 7}
    trunk_types = {'oak': 0, 'slime': 1, 'swamp': 2, 'cherry': 3, 'old': 4, 'jungle': 5}
    fruit_types = {'circle': 0, 'hanging': 1, 'berry': 2, 'long': 3, 'star': 4, 'pop': 5, 'fruitless': 6}

    def __init__(self, image_dir, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, ''))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_list[idx])).convert("RGBA")
        tensor_img = F.to_tensor(img)
        tensor_img = F.normalize(tensor_img, [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])

        tokens = self.image_list[idx].split('.')[0].split('_')
        fruit_color_idx = 7 if len(tokens) == 8 else 6
        feature = torch.tensor([
            PixelArtDataset.leaf_types[tokens[0]],  # Leaf Type Int
            int(tokens[4][0:2], 16),  # Leaf Color (R)
            int(tokens[4][2:4], 16),  # Leaf Color (G)
            int(tokens[4][4:6], 16),  # Leaf Color (B)
            PixelArtDataset.trunk_types[tokens[2]],  # Trunk Type Int
            PixelArtDataset.fruit_types[tokens[5]],  # Fruit Type Int
            int(tokens[fruit_color_idx][0:2], 16),  # Fruit Color (R)
            int(tokens[fruit_color_idx][2:4], 16),  # Fruit Color (G)
            int(tokens[fruit_color_idx][4:6], 16),  # Fruit Color (B)
        ], dtype=torch.int)

        return tensor_img, feature, self.image_list[idx]

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg', 'JPEG']:
                raise ValueError
