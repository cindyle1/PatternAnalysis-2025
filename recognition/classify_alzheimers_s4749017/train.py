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

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_model(
    model_save_path=MODEL_PATH,
    epochs=EPOCHS,
    lr=LR,
    weight_decay=WD
):
    """
    Trains the ConvNeXt model and saves the best checkpoint.

    Args:
        model_save_path (str): Where to save the best model.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for AdamW.
    """
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    model = ConvNeXt().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{epochs} | train {train_loss:.4f}/{train_acc:.3f} | val {val_loss:.4f}/{val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Saved best → {model_save_path} (val_acc={best_val_acc:.3f})")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")
    return model


if __name__ == "__main__":
    train_model()