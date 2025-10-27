import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from dataset import ADNIDataset, list_files

# keep modules.py unchanged; import your class name here
try:
    from modules import ConvNeXt as Net  
except Exception:
    try:
        from modules import ConvNeXtTiny2Class as Net
    except Exception:
        from modules import Model as Net

def pick_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    device = pick_device()
    items = list_files(args.data_dir)
    ds = ADNIDataset(items, image_size=args.image_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = Net(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    probs = []
    for x, _ in dl:
        x = x.to(device)
        logits = model(x)
        p1 = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        probs.extend(p1.tolist())

    print("Predictions (probability of AD):")
    for (path, _), p in list(zip(items, probs))[:10]:
        print(f"{os.path.basename(path)} -> {p:.3f}")

if __name__ == "__main__":
    main()
