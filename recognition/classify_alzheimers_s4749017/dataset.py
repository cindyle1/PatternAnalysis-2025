# dataset.py

import os
import random
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def make_loaders(
    root_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    val_size: float = 0.2,
    seed: int = 42,
    augment: bool = True,
):

    train_dir = os.path.join(root_dir, "train")
    test_dir  = os.path.join(root_dir, "test")

    # We'll treat images as 1-channel (grayscale MRI slices),
    # then normalize with ImageNet-ish stats for 1 channel.
    imagenet_mean_1c = (0.485,)
    imagenet_std_1c  = (0.229,)

    if augment:
        train_tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_1c, imagenet_std_1c),
        ])
    else:
        train_tfms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean_1c, imagenet_std_1c),
        ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean_1c, imagenet_std_1c),
    ])

    # 1. Load the full training dataset (will later be split into train/val)
    full_train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)

    # 2. Split into train/val (stratified-ish via manual seeding + random_split)
    n_total = len(full_train_ds)
    n_val = int(val_size * n_total)
    n_train = n_total - n_val

    random.seed(seed)
    # random_split uses torch.Generator for reproducibility, so we set that too
    import torch
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=g)

    # Make sure validation does NOT get augmentation
    # We can safely override the underlying dataset transform for the val subset.
    val_ds.dataset.transform = eval_tfms

    # 3. Test dataset (no augmentation)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    # 4. DataLoaders
    # num_workers=2 is fine on Colab; if you get DataLoader worker errors on Windows, set it to 0.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
