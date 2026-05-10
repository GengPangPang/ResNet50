import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Your datasets are all .jpg. Only scan .jpg files.
IMG_EXTENSIONS = (".jpg",)


def train_transform(image_size: int = 112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def eval_transform(image_size: int = 112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        image = Image.open(f)
        return image.convert("RGB")


def natural_key(name: str):
    """
    Natural sort key.

    Example:
        id_2 comes before id_10.
    """
    import re
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


class WebFaceFolder(Dataset):
    """
    Folder-style WebFace/CASIA-WebFace dataset.

    Expected:
        root/
          id_0/
            0001.jpg
            0002.jpg
          id_1/
            0001.jpg
            0002.jpg

    Label file is not required.
    Each subfolder is treated as one identity class.
    All .jpg images in every non-empty identity folder are used.

    No min_images_per_id filtering is applied.
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Training root not found: {root}")

        self.samples: List[Tuple[str, int]] = []
        self.class_to_idx = {}

        identities = sorted([
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        ], key=natural_key)

        if len(identities) == 0:
            raise RuntimeError(f"No identity folders found under: {root}")

        class_idx = 0
        skipped_empty_folders = []

        for identity in identities:
            identity_dir = os.path.join(root, identity)

            images = sorted([
                name for name in os.listdir(identity_dir)
                if name.lower().endswith(IMG_EXTENSIONS)
            ], key=natural_key)

            # Only skip completely empty folders.
            # Do not filter by min image count.
            if len(images) == 0:
                skipped_empty_folders.append(identity)
                continue

            self.class_to_idx[identity] = class_idx

            for image_name in images:
                image_path = os.path.join(identity_dir, image_name)
                self.samples.append((image_path, class_idx))

            class_idx += 1

        if not self.samples:
            raise RuntimeError(f"No valid .jpg training images found under: {root}")

        self.num_classes = len(self.class_to_idx)
        self.skipped_empty_folders = skipped_empty_folders

        if self.num_classes <= 1:
            raise RuntimeError(
                f"Need at least 2 identity classes for ArcFace training, got {self.num_classes}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = pil_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _lfw_path(lfw_root: str, name: str, index: int) -> str:
    return os.path.join(lfw_root, name, f"{name}_{index:04d}.jpg")


def read_lfw_pairs(pairs_path: str, lfw_root: str) -> List[Tuple[str, str, int]]:
    """
    Official LFW pairs.txt format:

    Header:
        10 300

    Same identity:
        Name index1 index2

    Different identities:
        Name1 index1 Name2 index2
    """
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"LFW pairs file not found: {pairs_path}")

    with open(pairs_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise RuntimeError(f"Empty LFW pairs file: {pairs_path}")

    start = 1 if len(lines[0].split()) == 2 and all(x.isdigit() for x in lines[0].split()) else 0

    pairs = []

    for line in lines[start:]:
        parts = line.replace("\t", " ").split()

        if len(parts) == 3:
            name, idx1, idx2 = parts
            img1 = _lfw_path(lfw_root, name, int(idx1))
            img2 = _lfw_path(lfw_root, name, int(idx2))
            is_same = 1
        elif len(parts) == 4:
            name1, idx1, name2, idx2 = parts
            img1 = _lfw_path(lfw_root, name1, int(idx1))
            img2 = _lfw_path(lfw_root, name2, int(idx2))
            is_same = 0
        else:
            raise ValueError(f"Invalid LFW pair line: {line}")

        pairs.append((img1, img2, is_same))

    return pairs


class LFWDataset(Dataset):
    def __init__(self, lfw_root: str, pairs_path: str, transform=None):
        self.lfw_root = lfw_root
        self.pairs_path = pairs_path
        self.transform = transform
        self.pairs = read_lfw_pairs(pairs_path, lfw_root)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img1_path, img2_path, is_same = self.pairs[index]

        if not os.path.isfile(img1_path):
            raise FileNotFoundError(f"LFW image not found: {img1_path}")
        if not os.path.isfile(img2_path):
            raise FileNotFoundError(f"LFW image not found: {img2_path}")

        img1 = pil_loader(img1_path)
        img2 = pil_loader(img2_path)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(is_same, dtype=torch.long)
