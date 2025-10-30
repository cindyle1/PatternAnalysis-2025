import os
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loaders


# hyperparameters / config
DATA_ROOT       = "/content/drive/MyDrive/AD_NC/"
MODEL_SIZE      = "SMALL"     # "TINY", "SMALL", "BASE"
EPOCHS          = 40
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-3
COMPILE         = False       # turn this OFF for now

# model configs
MODEL_CONFIG_TINY = {
    "in_chans": 20,
    "num_classes": 2,
    "depths": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768],
    # no drop_path_rate
}

MODEL_CONFIG_SMALL = {
    "in_chans": 20,
    "num_classes": 2,
    "depths": [3, 3, 27, 3],
    "dims": [96, 192, 384, 768],
    "drop_path_rate": 0.2,
}

MODEL_CONFIG_BASE = {
    "in_chans": 20,
    "num_classes": 2,
    "depths": [3, 3, 27, 3],
    "dims": [128, 256, 512, 1024],
}

IMAGE_SIZE = 224
if MODEL_SIZE == "TINY":
    MODEL_CONFIG = MODEL_CONFIG_TINY
    IMAGE_SIZE = 224
elif MODEL_SIZE == "SMALL":
    MODEL_CONFIG = MODEL_CONFIG_SMALL
    IMAGE_SIZE = 224
elif MODEL_SIZE == "BASE":
    MODEL_CONFIG = MODEL_CONFIG_BASE
    IMAGE_SIZE = 384

# --- import model class ---
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

    # dataloaders
    train_loader, val_loader, test_loader = make_loaders(
        root_dir=DATA_ROOT,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_size=0.2,
        seed=42,
        augment=True,
    )

    criterion = nn.CrossEntropyLoss()  # balanced case

    model = Net(**MODEL_CONFIG).to(device)

    # turn off compile for now (it was triggering dynamo weirdness before)
    # if COMPILE and hasattr(torch, "compile"):
    #     model = torch.compile(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    best_acc = 0.0
    best_state = None

    # training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0

        for batch_id, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        ):
            x, y = x.to(device), y.to(device)

            # one-time debug for first batch of first epoch
            if epoch == 1 and batch_id == 0:
                print("DEBUG x.shape:", x.shape)
                print("DEBUG x.min(), x.max():", x.min().item(), x.max().item())

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # gradient clipping -> helps stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            loss_sum += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_train_loss = loss_sum / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)

        # validation
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.3f}  "
            f"train_acc={train_acc*100:.2f}%  "
            f"val_acc={val_acc*100:.2f}%"
        )

        # track best checkpoint in RAM
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        # step LR scheduler
        scheduler.step()

    print("Best val acc:", best_acc * 100, "%")

    # reload best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # final test
    test_acc = evaluate(model, test_loader, device)
    print("Test acc:", test_acc * 100, "%")
    print("Training complete.")


if __name__ == "__main__":
    main()
