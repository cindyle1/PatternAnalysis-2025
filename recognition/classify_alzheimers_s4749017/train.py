import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score

from modules import ConvNeXt
from dataset import train_loader, val_loader, test_loader

# CONFIG
EPOCHS          = 15
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-2
MAX_GRAD_NORM   = 1.0
EARLY_PATIENCE  = 10
MODEL_PATH      = "./models/best_convnext.pth"
FIG_DIR         = "./figures"
DEVICE          = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
NUM_CLASSES     = 2
IN_CHANS        = 1

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# HELPERS
@torch.no_grad()
def evaluate_full(model, loader, device):
    """
    Returns:
      acc_overall (float 0..1)
      bal_acc (float 0..1)  <-- mean recall across classes
      per_class_recall {cls: recall (0..1)}
    """
    model.eval()
    ys = []
    yh = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(1)

        ys.extend(y.cpu().tolist())
        yh.extend(preds.cpu().tolist())

    acc_overall = accuracy_score(ys, yh)

    recalls = recall_score(ys, yh, average=None, labels=list(range(NUM_CLASSES)))
    per_class_recall = {cls: float(recalls[cls]) for cls in range(len(recalls))}
    bal_acc = float(recalls.mean())

    return acc_overall, bal_acc, per_class_recall


@torch.no_grad()
def evaluate_test_breakdown(model, loader, device):
    """
    Final testing: overall acc + per-class acc.
    """
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {}
    per_class_total = {}

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(1)

        correct += (preds == y).sum().item()
        total += y.size(0)

        for cls in range(NUM_CLASSES):
            mask = (y == cls)
            per_class_total[cls]   = per_class_total.get(cls, 0)  + mask.sum().item()
            per_class_correct[cls] = per_class_correct.get(cls, 0) + ((preds == y) & mask).sum().item()

    overall = correct / max(total, 1)
    per_class_acc = {
        cls: (per_class_correct.get(cls, 0) / max(per_class_total.get(cls, 1), 1))
        for cls in per_class_total
    }
    return overall, per_class_acc


def get_class_weights(loader, num_classes):
    """
    Estimate class weights for CrossEntropyLoss from label frequency in the given loader.
    Normalised so mean(weight) ~ 1.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, y in loader:
        for c in range(num_classes):
            counts[c] += (y == c).sum().item()

    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return weights


def plot_curves(train_losses, val_bal_accs, out_dir=FIG_DIR):
    """
    Save final curves at the end of training.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "final_loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot([a * 100.0 for a in val_bal_accs], label="Val Balanced Acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy (%)")
    plt.title("Validation Balanced Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "final_accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"✅ Saved final plots to:\n - {loss_path}\n - {acc_path}")


# TRAIN
def main():
    print("Device:", DEVICE)

    # build model
    model = ConvNeXt(in_chans=IN_CHANS, num_classes=NUM_CLASSES).to(DEVICE)

    # mixed precision on CUDA only
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # class-weighted loss for imbalance robustness
    class_weights = get_class_weights(train_loader, NUM_CLASSES).to(DEVICE)
    print("Class weights for loss (normalised inverse freq):", class_weights.cpu().tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    best_val_bal_acc = 0.0
    best_state = None
    epochs_since_improve = 0

    train_losses = []
    val_bal_accs = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # --- training step loop ---
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)

            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        # ---- epoch-level metrics on train ----
        avg_train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ALSO compute train balanced acc with evaluate_full
        train_acc_full, train_bal_acc_full, train_recalls = evaluate_full(model, train_loader, DEVICE)

        # ---- validation ----
        val_acc, val_bal_acc, val_recalls = evaluate_full(model, val_loader, DEVICE)

        train_losses.append(avg_train_loss)
        val_bal_accs.append(val_bal_acc)

        print(
            f"Epoch {epoch}: "
            f"train_loss={avg_train_loss:.3f}  "
            f"train_acc={train_acc*100:.2f}%  "
            f"train_bal_acc={train_bal_acc_full*100:.2f}%  "
            f"val_acc={val_acc*100:.2f}%  "
            f"val_bal_acc={val_bal_acc*100:.2f}%  "
            f"val_recalls={ {cls:f'{rec:.2f}' for cls,rec in val_recalls.items()} }"
        )

        # checkpoint on balanced acc
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0

            torch.save({"model_state": best_state}, MODEL_PATH)
            print(f"  ✓ New best val_bal_acc={best_val_bal_acc*100:.2f}%. Saved {MODEL_PATH}")
        else:
            epochs_since_improve += 1

        scheduler.step()

        if epochs_since_improve >= EARLY_PATIENCE:
            print(f"  ⏹ Early stopping: no val_bal_acc improvement for {EARLY_PATIENCE} epochs.")
            break

    print("Best val balanced acc:", best_val_bal_acc * 100, "%")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    test_acc_overall, test_breakdown = evaluate_test_breakdown(model, test_loader, DEVICE)
    print("Test acc:", test_acc_overall * 100, "%")
    print("Per-class test acc:", {cls: round(acc*100, 2) for cls, acc in test_breakdown.items()})

    print("Training complete.")

    plot_curves(train_losses, val_bal_accs)

    print(f"All done. Best model weights are in {MODEL_PATH}")
    print(f"Training curves saved under {FIG_DIR}/")


if __name__ == "__main__":
    main()
