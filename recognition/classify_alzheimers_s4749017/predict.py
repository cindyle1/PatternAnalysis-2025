#test accuracies and confusion matrix for performance

import torch, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from modules import ConvNeXt
from dataset import test_loader

MODEL_PATH = "./models/convnext_tiny_best.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"