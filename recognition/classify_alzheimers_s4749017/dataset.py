#confirm data is loaded in correctly

#SET UP IMPORTS
import os, glob
from typing import List, Tuple
import numpy as np
import nibabel as nib
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_TO_IDX = {"CN": 0, "AD": 1}

def list_files(root: str) -> List[Tuple[str, int]]:
    items = []
    for cls in CLASS_TO_IDX:
        cls_dir = os.path.join(root, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.nii"))) + \
                sorted(glob.glob(os.path.join(cls_dir, "*.nii.gz")))
        for f in files:
            items.append((f, CLASS_TO_IDX[cls]))
    if not items:
        raise FileNotFoundError(
            f"No NIfTI files under {root}. Expected {root}/AD/*.nii.gz and {root}/CN/*.nii.gz"
        )
    return items

from PIL import Image
import numpy as np

def three_views_rgb(vol_xyz: np.ndarray) -> np.ndarray:
    """
    vol: (X,Y,Z)
    Extract mid axial/coronal/sagittal slices, resize them to the same shape
    using PIL (no cv2), normalize to [0,1], and stack to (3,H,W).
    """
    X, Y, Z = vol_xyz.shape
    axial   = vol_xyz[:, :, Z // 2]
    coronal = vol_xyz[:, Y // 2, :]
    sagittal = vol_xyz[X // 2, :, :]

    slices = [axial, coronal, sagittal]
    slices = [(s - s.min()) / (s.max() - s.min() + 1e-6) for s in slices]

    # Make all slices same size via PIL resize
    h_max = max(s.shape[0] for s in slices)
    w_max = max(s.shape[1] for s in slices)
    resized = []
    for s in slices:
        pil = Image.fromarray((s * 255).astype(np.uint8))
        pil = pil.resize((w_max, h_max), Image.BILINEAR)
        resized.append(np.array(pil, dtype=np.float32) / 255.0)

    img = np.stack(resized, axis=0)  # (3,H,W)
    return img



class ADNIDataset(Dataset):
    def __init__(self, items: List[Tuple[str,int]], image_size=224, augment=False):
        self.items = items
        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            (T.RandomHorizontalFlip(p=0.5) if augment else T.Lambda(lambda x: x)),
            T.ToTensor(),
            # ImageNet normalization (as taught for transfer learning)
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = np.nan_to_num(vol)
        img = three_views_rgb(vol)
        pil = Image.fromarray((np.transpose(img, (1,2,0)) * 255).astype(np.uint8))
        x = self.tf(pil)
        if x.shape[0] == 3:           # model expects 1 channel
            x = x.mean(dim=0, keepdim=True)
        return x, y


def make_split(root: str, test_size=0.2, val_size=0.1, seed=42):
    """
    Robust two-step split:
    1) Train vs Temp where Temp = (val + test) with size (val_size + test_size)
    2) Split Temp into Val and Test with proportions matching val_size : test_size
    """
    items = list_files(root)
    X = [p for p, _ in items]
    y = [l for _, l in items]

    # Step 1: carve out a combined temp set (val+test)
    temp_size = val_size + test_size
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=seed
    )

    # Step 2: split temp into val and test with the right ratio
    # (avoid tiny val set that breaks stratification)
    if temp_size == 0:
        X_val, y_val = [], []
        X_test, y_test = [], []
    else:
        # fraction of temp that should go to VALIDATION
        val_frac_within_temp = val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - val_frac_within_temp),  # keep val_frac for val
            stratify=y_temp,
            random_state=seed
        )

    pack = lambda xs, ys: list(zip(xs, ys))
    return pack(X_train, y_train), pack(X_val, y_val), pack(X_test, y_test)
