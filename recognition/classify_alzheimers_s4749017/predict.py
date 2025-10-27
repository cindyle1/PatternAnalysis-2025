#test accuracies and confusion matrix for performance

import torch, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from modules import ConvNeXt
from dataset import test_loader

MODEL_PATH = "./models/convnext_tiny_best.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model = ConvNeXt().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    ys, preds = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            preds.append(p)
            ys.append(y.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=["NC","AD"], digits=3))

if __name__ == "__main__":
    main()