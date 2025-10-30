import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import make_loaders

# --------------------------
# Inline "parameters" (no external file)
# --------------------------

# DATA_ROOT = "./data/AD_NC/"   
DATA_ROOT = "/content/drive/MyDrive/AD_NC/"

MODEL_SIZE = "SMALL"          # "SMALL", "TINY", "BASE"

BATCH_SIZE     = 16
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-3
EPOCHS         = 15
COMPILE        = True

MODEL_CONFIG_TINY = {
    "in_chans": 1,
    "num_classes": 2,
    "depths": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768],
}

MODEL_CONFIG_SMALL = {
    "in_chans": 1,
    "num_classes": 2,
    "depths": [3, 3, 27, 3],
    "dims": [96, 192, 384, 768],
    "drop_path_rate": 0.2,
}

MODEL_CONFIG_BASE = {
    "in_chans": 1,
    "num_classes": 2,
    "depths": [3, 3, 27, 3],
    "dims": [128, 256, 512, 1024],
}


IMAGE_SIZE = 224
MODEL_CONFIG = MODEL_CONFIG_TINY

if MODEL_SIZE == "TINY":
    MODEL_CONFIG = MODEL_CONFIG_TINY
    IMAGE_SIZE = 224
elif MODEL_SIZE == "SMALL":
    MODEL_CONFIG = MODEL_CONFIG_SMALL
    IMAGE_SIZE = 224
elif MODEL_SIZE == "BASE":
    MODEL_CONFIG = MODEL_CONFIG_BASE
    IMAGE_SIZE = 384


# --- Model import ---
try:
    from modules import ConvNeXt as Net
except Exception:
    try:
        from modules import ConvNeXtTiny2Class as Net
    except Exception:
        from modules import Model as Net


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, yh = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        ys.extend(y.cpu().numpy())
        yh.extend(pred.cpu().numpy())
    return accuracy_score(ys, yh)


def main():
    device = pick_device()
    print("Device:", device)

    # --- Build dataloaders using constants above ---
    train_loader, val_loader, test_loader = make_loaders(
        root_dir=DATA_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_size=0.2,
        seed=42,
        augment=True,
    )

    # --- Build model with chosen MODEL_CONFIG ---
    model = Net(**MODEL_CONFIG).to(device)

    if COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    # loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_acc = 0.0
    best_state = None

    # --- Training loop ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_acc = evaluate(model, val_loader, device)
        avg_loss = loss_sum / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.3f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print("Best val acc:", best_acc * 100, "%")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_acc = evaluate(model, test_loader, device)
    print("Test acc:", test_acc * 100, "%")
    print("✅ Training complete (no files saved).")


if __name__ == "__main__":
    main()
