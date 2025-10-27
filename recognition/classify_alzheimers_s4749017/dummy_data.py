# dummy dataset
import os, numpy as np, nibabel as nib, pathlib, argparse
rng = np.random.default_rng(0)

def make_split(root, cls, n=12, shape=(96,96,64), signal=False):
    d = pathlib.Path(root)/cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        x = rng.normal(0, 1, shape).astype(np.float32)
        if signal:
            # carve a brighter blob to mimic a "pattern"
            xx, yy, zz = np.indices(shape)
            cx, cy, cz = [s//2 for s in shape]
            mask = (xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2 < (min(shape)//5)**2
            x[mask] += 3.0
        nib.Nifti1Image(x, affine=np.eye(4)).to_filename(str(d/f"subj_{i:03d}.nii.gz"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    make_split(args.out, "CN", n=24, signal=False)
    make_split(args.out, "AD", n=24, signal=True)
    print("Dummy ADNI created at", args.out)
