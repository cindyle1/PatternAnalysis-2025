#confirm data is loaded in correctly

#SET UP IMPORTS
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os, torch, random

#SET UP PARAMS
DATA_ROOT = "./data/AD_NC/"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
IMAGE_SIZE = 224
BATCH_SIZE = 16
IMAGENET_MEAN_1C, IMAGENET_STD_1C = (0.485,), (0.229,)

#TRANSFORMS
train_tfms = transforms.Compose([
    transforms.Grayscale(1), 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN_1C, IMAGENET_STD_1C),
])

evaluate_tfms = transforms.Compose([
    transforms.Grayscale(1), 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN_1C, IMAGENET_STD_1C),
])

#DATASETS
train_full = datasets.ImageFolder(TRAIN_DIR, transform = train_tfms)
train_set = datasets.ImageFolder(TEST_DIR, transforms = evaluate_tfms)

#Random 80/20 split
val_ratio = 0.2
n_total = len(train_full)
c_val = int(n_total * val_ratio)
n_train = n_total - n_val
g = torch.Generator().manual_seed(42)
train_set, val_set = random_split(train_full, [n_train, n_val], generator=g)

# LOADERS
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)