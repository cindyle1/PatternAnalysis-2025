# # dataset.py

import os
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler  # ### CHANGED (import sampler)
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIG 
DATA_ROOT = "./data/AD_NC/"
# DATA_ROOT = "/content/drive/MyDrive/AD_NC/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

IMAGE_SIZE = 224
BATCH_SIZE = 16
VAL_RATIO  = 0.2
SEED       = 42
NUM_WORKERS = 2
PIN_MEMORY  = True

# UTILITIES 
def normalise_intensity(img: Image.Image) -> Image.Image:
    """
    Convert PIL image -> numpy -> intensity normalize (z-score then min-max to 0-255).
    Keeps anatomical contrast but reduces scanner bias.
    """
    arr = np.array(img).astype(np.float32)
    mean, std = arr.mean(), arr.std()
    arr = (arr - mean) / (std + 1e-5)                           # z-score
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)    # scale 0–1
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_subject_id(path: str) -> str:
    """
    Extract subject ID from filename.
    Assumes filenames look like '218391_79.jpeg' -> subject '218391'.
    """
    base = os.path.basename(path)
    return base.split("_")[0]

# TRANSFORMS 
train_tfms = transforms.Compose([
    transforms.Lambda(normalise_intensity),
    transforms.Grayscale(1),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

evaluate_tfms = transforms.Compose([
    transforms.Lambda(normalise_intensity),
    transforms.Grayscale(1),
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

dataset_train_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
dataset_eval_aug  = datasets.ImageFolder(TRAIN_DIR, transform=evaluate_tfms)

# test set ALWAYS eval tfms
test_set = datasets.ImageFolder(TEST_DIR, transform=evaluate_tfms)

# SUBJECT-WISE TRAIN/VAL SPLIT
# build per-image info from the *eval* view (deterministic transform for IDs/labels)
all_image_paths = [p for (p, _) in dataset_eval_aug.samples]
all_labels      = [y for (_, y) in dataset_eval_aug.samples]
all_subjects    = [get_subject_id(p) for p in all_image_paths]

df = pd.DataFrame({
    "idx":      list(range(len(all_image_paths))),
    "subject":  all_subjects,
    "label":    all_labels,
})

# collapse to 1 row per subject
subject_df = df.groupby("subject").first().reset_index()
# columns: ['subject', 'idx', 'label']

# subject-level stratified split
train_subjects, val_subjects = train_test_split(
    subject_df["subject"],
    test_size=VAL_RATIO,
    random_state=SEED,
    stratify=subject_df["label"],
)

train_subjects = set(train_subjects)
val_subjects   = set(val_subjects)

# back to slice indices
train_indices = df[df["subject"].isin(train_subjects)]["idx"].tolist()
val_indices   = df[df["subject"].isin(val_subjects)]["idx"].tolist()

# subsets
train_set = Subset(dataset_train_aug, train_indices) 
val_set   = Subset(dataset_eval_aug,  val_indices)  

# BUILD WEIGHTED SAMPLER FOR TRAIN 
# We need the label for each sample in train_set, in the same order as train_indices.
# dataset_train_aug.samples[orig_idx] = (filepath, class_idx)
train_labels = []
for orig_idx in train_indices:
    _, cls = dataset_train_aug.samples[orig_idx]
    train_labels.append(cls)

train_labels = np.array(train_labels)

# inverse frequency weighting
class_sample_counts = np.bincount(train_labels)       # e.g. [num_class0, num_class1]
class_weights = 1.0 / class_sample_counts             # weight per class
sample_weights = class_weights[train_labels]          # map each sample -> weight

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

# LOADERS

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    sampler=sampler,  
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

# ---------------- DEBUG / INFO PRINT ----------------
print(f"[dataset] Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")
print(f"[dataset] Classes={dataset_train_aug.classes}")
print(f"[dataset] Example subjects in train: {list(list(train_subjects)[:3])}")
print(f"[dataset] Example subjects in val:   {list(list(val_subjects)[:3])}")
