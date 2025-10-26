#leaning process

import os, torch
from torch import nn, optim
from modules import ConvNeXt
from dataset import train_loader, val_loader

MODEL_PATH = "./models/convnext_tiny_best.pth"
EPOCHS = 10
LR = 3e-4
WD = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

