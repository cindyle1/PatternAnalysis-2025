# dataset.py

import os
import sys
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# --------------------------
# Inline "parameters"
# --------------------------
DATA_ROOT        = "./data/AD_NC/"   # you can change this when you run train.py if needed
RANDOM_STATE     = 42
NORMALISATION_M  = 0.1114
NORMALISATION_SD = 0.2184
CHANNELS         = 1         # per slice (grayscale)
IMAGE_SIZE       = 224       # may get overridden in train.py if you choose BASE
VAL_FRACTION     = 0.2       # % of train used as validation

class PatientDataset(Dataset):
    """
    One item = one patient.
    We return ONE "representative" slice for that patient as a (1,H,W) tensor.
    """

    def __init__(
        self,
        patient_ids: List[str],
        pid_to_slices: Dict[str, List[str]],
        pid_to_label: Dict[str, int],
        transform: transforms.Compose,
        representative_only: bool = True,
    ):
        self.patient_ids = patient_ids
        self.pid_to_slices = pid_to_slices
        self.pid_to_label = pid_to_label
        self.transform = transform
        self.representative_only = representative_only

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        slice_paths = self.pid_to_slices[pid]
        label = self.pid_to_label[pid]

        if self.representative_only:
            mid_idx = len(slice_paths) // 2
            img = Image.open(slice_paths[mid_idx]).convert("L")  # grayscale
            x = self.transform(img)  # (1,H,W)
            return x, label
        else:
            # (not used right now, but kept for completeness)
            slices = []
            for sp in slice_paths:
                img = Image.open(sp).convert("L")
                x = self.transform(img)  # (1,H,W)
                x = x.squeeze(0)         # -> (H,W)
                slices.append(x)
            volume = torch.stack(slices, dim=0)  # (num_slices,H,W)
            return volume, label


def _build_patient_maps(samples: List[Tuple[str, int]]):
    """
    Turn ImageFolder file list into patient_id -> [slice_paths]
    Assumes filenames like "<patientID>_something.png"
    """
    pid_to_slices = defaultdict(list)
    pid_to_label = {}

    for path, label in samples:
        fname = os.path.basename(path)
        pid = fname.split("_")[0]  # <- adjust this rule if naming differs
        pid_to_slices[pid].append(path)
        pid_to_label[pid] = label

    for pid in pid_to_slices:
        pid_to_slices[pid] = sorted(pid_to_slices[pid])

    return pid_to_slices, pid_to_label


def _make_transforms(image_size: int, augment: bool):
    """
    image_size: resize target (224 normally / 384 for BASE config)
    NORMALISATION_M, NORMALISATION_SD: given stats for normalisation
    """

    norm_mean = [NORMALISATION_M]
    norm_std  = [NORMALISATION_SD]

    eval_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
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
                scale=(0.95, 1.05)
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15
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
        print("[dataset.py] ✅ no patient overlap between train / val / test")


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
    root_dir must contain:
        root_dir/train/<class>/*.png
        root_dir/test/<class>/*.png
    """
    train_dir = os.path.join(root_dir, "train")
    test_dir  = os.path.join(root_dir, "test")

    # 1. load raw slice paths/labels
    trainval_folder = datasets.ImageFolder(root=train_dir, transform=None)
    test_folder     = datasets.ImageFolder(root=test_dir,  transform=None)

    class_to_idx = trainval_folder.class_to_idx
    print(f"[dataset.py] class_to_idx = {class_to_idx}")

    # 2. group slices -> patients
    trainval_pid_to_slices, trainval_pid_to_label = _build_patient_maps(
        trainval_folder.samples
    )
    test_pid_to_slices, test_pid_to_label = _build_patient_maps(
        test_folder.samples
    )

    trainval_pids = list(trainval_pid_to_slices.keys())
    test_pids     = list(test_pid_to_slices.keys())

    # 3. stratified split on patient IDs to get train vs val
    train_pids, val_pids = train_test_split(
        trainval_pids,
        test_size=val_size,
        random_state=seed,
        stratify=[trainval_pid_to_label[pid] for pid in trainval_pids],
    )

    print(f"[dataset.py] #train patients: {len(train_pids)}")
    print(f"[dataset.py] #val patients:   {len(val_pids)}")
    print(f"[dataset.py] #test patients:  {len(test_pids)}")

    _leak_check(train_pids, val_pids, test_pids)

    # 4. build transforms
    train_tfm, eval_tfm = _make_transforms(image_size=image_size, augment=augment)

    # We stick to ONE slice per patient
    representative_only = True

    train_dataset = PatientDataset(
        patient_ids=train_pids,
        pid_to_slices=trainval_pid_to_slices,
        pid_to_label=trainval_pid_to_label,
        transform=train_tfm,
        representative_only=representative_only,
    )

    val_dataset = PatientDataset(
        patient_ids=val_pids,
        pid_to_slices=trainval_pid_to_slices,
        pid_to_label=trainval_pid_to_label,
        transform=eval_tfm,
        representative_only=representative_only,
    )

    test_dataset = PatientDataset(
        patient_ids=test_pids,
        pid_to_slices=test_pid_to_slices,
        pid_to_label=test_pid_to_label,
        transform=eval_tfm,
        representative_only=representative_only,
    )

    # 5. data loaders (NO saving / NO new dirs)
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
