import os, argparse, time
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ADNIDataset, make_split, list_files

DATA_ROOT = "/content/drive/MyDrive/AD_NC"

try:
    from modules import ConvNeXt as Net  # ← change this one line if your class is named differently
except Exception:
    # gentle fallbacks if your class has another name
    try:
        from modules import ConvNeXtTiny2Class as Net
    except Exception:
        from modules import Model as Net  # last resort if you named it 'Model'

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
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
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    print("Device:", device)

    train_dir = os.path.join(args.data_dir, "train")
    test_dir  = os.path.join(args.data_dir, "test")

    tr, va, _ = make_split(train_dir, test_size=0.0, val_size=0.2, seed=42)
    te = list_files(test_dir)

    train_ds = ADNIDataset(tr, image_size=args.image_size, augment=args.augment)
    val_ds   = ADNIDataset(va, image_size=args.image_size)
    test_ds  = ADNIDataset(te, image_size=args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        model = Net(pretrained=args.pretrained).to(device)
    except TypeError:
        model = Net().to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={loss_sum/len(train_loader):.3f}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            #keep in cpu instead of saving locally
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        

    print("Best val acc:", best_acc)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device) 

    test_acc = evaluate(model, test_loader, device)
    print("Test acc:", test_acc)


if __name__ == "__main__":
    main()
