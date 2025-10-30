import os, argparse, time
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import make_loaders 

# Try to import your model class
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Root dir that contains train/ and test/ subfolders")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--out_dir", default="outputs",
                    help="Where to save best_model.pth")
    args = ap.parse_args()

    device = pick_device()
    print("Device:", device)

    # Create loaders from your folder structure
    train_loader, val_loader, test_loader = make_loaders(
        root_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=0.2,
        seed=42,
        augment=args.augment,
    )

    # Build model
    try:
        model = Net(pretrained=args.pretrained).to(device)
    except TypeError:
        # fallback if Net() doesn't accept pretrained=
        model = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for x, y in tqdm(train_loader,
                         desc=f"Epoch {epoch}/{args.epochs}",
                         leave=False):
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

        # track best weights
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print("Best val acc:", best_acc * 100, "%")

    # Load best weights back into the model before final test
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Final test accuracy
    test_acc = evaluate(model, test_loader, device)
    print("Test acc:", test_acc * 100, "%")

    # Save best checkpoint so predict.py can load it later
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "best_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    main()
