"""Dataset helpers.

Defaults to **STL-10** at 64x64 RGB: a real-world photo dataset of 10 classes
(airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) shipped at
96x96 by torchvision and resized to 64x64 here. STL-10 also includes a large
unlabeled split (100k images) which we use for training - perfect for a VAE,
which does not need labels.

The images are intentionally low-res; the user plans to feed VAE outputs into
an upsampling model, so 64x64 is a deliberate target.

Other available choices:
  - "cifar10":  32x32 RGB photos, upscaled to 64x64
  - "mnist" / "fashion": 28x28 grayscale (legacy)
"""

from __future__ import annotations

import os
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


def _rgb_tfm(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0,1] floats, shape (3,H,W)
    ])


def get_dataloaders(name: str = "stl10", data_root: str = "./data",
                    batch_size: int = 128, num_workers: int = 0,
                    image_size: int = 64):
    """Returns (train_loader, test_loader, in_channels, image_size)."""
    name = name.lower()
    os.makedirs(data_root, exist_ok=True)

    if name == "stl10":
        tfm = _rgb_tfm(image_size)
        # `unlabeled` is a 100k split of real photos - ideal for VAE training.
        unlabeled = datasets.STL10(data_root, split="unlabeled",
                                   download=True, transform=tfm)
        train_lbl = datasets.STL10(data_root, split="train",
                                   download=True, transform=tfm)
        train = ConcatDataset([unlabeled, train_lbl])
        test  = datasets.STL10(data_root, split="test",
                               download=True, transform=tfm)
        in_channels = 3
    elif name == "cifar10":
        tfm = _rgb_tfm(image_size)
        train = datasets.CIFAR10(data_root, train=True,  download=True, transform=tfm)
        test  = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
        in_channels = 3
    elif name == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(data_root, train=True,  download=True, transform=tfm)
        test  = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        in_channels, image_size = 1, 28
    elif name == "fashion":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(data_root, train=True,  download=True, transform=tfm)
        test  = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
        in_channels, image_size = 1, 28
    else:
        raise ValueError(f"Unknown dataset: {name}")

    pin = num_workers > 0
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                              pin_memory=pin)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader, in_channels, image_size
