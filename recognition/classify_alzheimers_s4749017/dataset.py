# dataset.py

import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image


# Inline parameters / defaults
RANDOM_STATE     = 42
NORMALISATION_M  = 0.1114
NORMALISATION_SD = 0.2184
IMAGE_SIZE       = 224   # BASE model can bump this to 384


class PatientDataset(Dataset):
    """
    Returns one tensor per patient:
      x: (20, H, W)  <-- 20 slices stacked as channels
      y: int label
    This matches a ConvNeXt where in_chans=20.
    """
    def __init__(
        self,
        patient_ids: List[str],
        pid_to_slices: Dict[str, List[str]],
        pid_to_label: Dict[str, int],
        transform: transforms.Compose,
        num_slices: int = 20,
    ):
        self.patient_ids = patient_ids
        self.pid_to_slices = pid_to_slices
        self.pid_to_label = pid_to_label
        self.transform = transform
        self.num_slices = num_slices

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        slice_paths = self.pid_to_slices[pid]
        label = self.pid_to_label[pid]

        # Ensure exactly num_slices slices per patient by trimming/padding
        N = self.num_slices
        if len(slice_paths) >= N:
            slice_paths = slice_paths[:N]
        else:
            slice_paths = slice_paths + [slice_paths[-1]] * (N - len(slice_paths))

        slices = []
        for path in slice_paths:
            img = Image.open(path).convert("L")  # grayscale
            x = self.transform(img)             # (1, H, W)
            slices.append(x)

        # Stack along channel dim -> (N, H, W)
        volume = torch.cat(slices, dim=0)
        return volume, label


def _build_patient_maps(samples: List[Tuple[str, int]]):
    """
    Converts a flat list of (image_path, class_idx) pairs into:
      - pid_to_slices: { patient_id: [slice_path1, slice_path2, ...] }
      - pid_to_label:  { patient_id: class_idx }
    Assumes filenames look like "<patientID>_whatever.png"
    """
    pid_to_slices = defaultdict(list)
    pid_to_label = {}

    for path, label in samples:
        fname = os.path.basename(path)
        pid = fname.split("_")[0]  # <- adjust if your naming scheme differs
        pid_to_slices[pid].append(path)
        pid_to_label[pid] = label

    # keep slices in a stable order
    for pid in pid_to_slices:
        pid_to_slices[pid] = sorted(pid_to_slices[pid])

    return pid_to_slices, pid_to_label


def _make_transforms(image_size: int, augment: bool):
    """
    Returns (train_transform, eval_transform).
    Normalises using provided dataset stats so values end ~[-1,1].
    """
    norm_mean = [NORMALISATION_M]
    norm_std  = [NORMALISATION_SD]

    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    if augment:
        train_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.RandomErasing(
                p=0.1,
                scale=(0.02, 0.05),
                value=0.0,
            ),
        ])
    else:
        train_tfm = eval_tfm

    return train_tfm, eval_tfm


def _leak_check(train_ids, val_ids, test_ids):
    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    leak_tv = train_set & val_set
    leak_tt = train_set & test_set
    leak_vt = val_set   & test_set

    if leak_tv or leak_tt or leak_vt:
        print("[dataset.py][ERROR] Patient ID leakage across splits!")
        print(f"train ∩ val  = {leak_tv}")
        print(f"train ∩ test = {leak_tt}")
        print(f"val ∩ test   = {leak_vt}")
        sys.exit(1)
    else:
        print("[dataset.py] no patient overlap between train / val / test")

def _count_labels(pids, pid_to_label):
    counts = {}
    for pid in pids:
        lab = pid_to_label[pid]
        counts[lab] = counts.get(lab, 0) + 1
    return counts

def make_loaders(
    root_dir: str,
    image_size: int,
    batch_size: int,
    val_size: float,
    seed: int,
    augment: bool,
):
    """
    Build train/val/test DataLoaders using patient-level splits.
    Expects:
        root_dir/train/<class>/*.png
        root_dir/test/<class>/*.png
    where multiple slices from the same patient share the same <patientID>_ prefix.
    """

    train_dir = os.path.join(root_dir, "train")
    test_dir  = os.path.join(root_dir, "test")

    # 1. Load flat image-level datasets (no transform yet)
    trainval_folder = datasets.ImageFolder(root=train_dir, transform=None)
    test_folder     = datasets.ImageFolder(root=test_dir,  transform=None)

    class_to_idx = trainval_folder.class_to_idx
    print(f"[dataset.py] class_to_idx = {class_to_idx}")

    # 2. Group all slices by patient ID
    trainval_pid_to_slices, trainval_pid_to_label = _build_patient_maps(
        trainval_folder.samples
    )
    test_pid_to_slices, test_pid_to_label = _build_patient_maps(
        test_folder.samples
    )

    trainval_pids = list(trainval_pid_to_slices.keys())
    test_pids     = list(test_pid_to_slices.keys())

    # 3. Split patient IDs into train / val (stratified by label)
    train_pids, val_pids = train_test_split(
        trainval_pids,
        test_size=val_size,
        random_state=seed,
        stratify=[trainval_pid_to_label[pid] for pid in trainval_pids],
    )

    print(f"[dataset.py] #train patients: {len(train_pids)}")
    print(f"[dataset.py] #val patients:   {len(val_pids)}")
    print(f"[dataset.py] #test patients:  {len(test_pids)}")

    print("[dataset.py] train label counts:", _count_labels(train_pids, trainval_pid_to_label))
    print("[dataset.py] val   label counts:", _count_labels(val_pids, trainval_pid_to_label))
    print("[dataset.py] test  label counts:", _count_labels(test_pids, test_pid_to_label))


    _leak_check(train_pids, val_pids, test_pids)

    # 4. Transforms
    train_tfm, eval_tfm = _make_transforms(image_size=image_size, augment=augment)

    # 5. Datasets (each item is a 20-slice volume, shape (20,H,W))
    train_dataset = PatientDataset(
        patient_ids=train_pids,
        pid_to_slices=trainval_pid_to_slices,
        pid_to_label=trainval_pid_to_label,
        transform=train_tfm,
        num_slices=20,
    )

    val_dataset = PatientDataset(
        patient_ids=val_pids,
        pid_to_slices=trainval_pid_to_slices,
        pid_to_label=trainval_pid_to_label,
        transform=eval_tfm,
        num_slices=20,
    )

    test_dataset = PatientDataset(
        patient_ids=test_pids,
        pid_to_slices=test_pid_to_slices,
        pid_to_label=test_pid_to_label,
        transform=eval_tfm,
        num_slices=20,
    )

    # 6. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
