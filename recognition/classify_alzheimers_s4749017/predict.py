# predict.py
import os, argparse, torch
import numpy as np
from dataset import make_loaders

# --- Model import with fallbacks ---
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
def main():
    # --- Parse CLI arguments ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root directory containing train/ and test/ folders")
    ap.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (.pth)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    # --- Set device ---
    device = pick_device()
    print(f"Using device: {device}")

    # --- Load data ---
    _, _, test_loader = make_loaders(
        root_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_size=0.2,
        seed=42,
        augment=False,
    )

    # --- Load model ---
    model = Net(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # --- Evaluate on test data ---
    all_probs, all_preds, all_labels = [], [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1 (AD)
        preds = logits.argmax(dim=1)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.numpy().tolist())

    # --- Compute accuracy ---
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

    # --- Print sample predictions ---
    print("\nSample predictions (probability of AD):")
    class_names = test_loader.dataset.classes
    for i in range(min(10, len(all_probs))):
        print(f"Sample {i+1}: Predicted={class_names[all_preds[i]]}, True={class_names[all_labels[i]]}, P(AD)={all_probs[i]:.3f}")


if __name__ == "__main__":
    main()
