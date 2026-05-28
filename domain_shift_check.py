"""
Cross-dataset domain shift check.
Runs inference on G1020 and ORIGA (both excluded from val/test) and reports
predicted glaucoma rate vs true rate. A well-calibrated model should be close.
"""
import os
import sys
import numpy as np
import torch
import timm
from torch.amp import autocast
from PIL import Image

sys.path.insert(0, r"C:\Projects\Glaucoma Project")
from data_pipeline import build_combined_df, val_test_transform

BASE   = r"C:\Projects\Glaucoma Project"
CKPT   = os.path.join(BASE, "models", "best_model.pth")
DEVICE = torch.device("cuda")

YOUDEN_THR = 0.3853   # from evaluate.py


def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_dataset(model, df):
    probs, labels = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(row["img_path"], row["filename"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        tensor = val_test_transform(image).unsqueeze(0).to(DEVICE)
        with autocast("cuda"):
            logit = model(tensor)
        p = float(torch.sigmoid(logit).squeeze().cpu())
        probs.append(p)
        labels.append(int(row["label"]))
    return np.array(labels), np.array(probs)


def report(name, labels, probs, thr=YOUDEN_THR):
    true_rate = labels.mean() * 100
    pred_rate = (probs >= thr).mean() * 100
    mean_prob = probs.mean() * 100
    print(f"\n  {name}")
    print(f"    N                   : {len(labels)}")
    print(f"    True glaucoma rate  : {true_rate:.1f}%")
    print(f"    Predicted rate      : {pred_rate:.1f}%  (thr={thr:.3f})")
    print(f"    Mean P(glaucoma)    : {mean_prob:.1f}%")
    gap = abs(pred_rate - true_rate)
    status = "OK" if gap < 15 else "SHIFTED"
    print(f"    Gap                 : {gap:.1f}pp  [{status}]")


def main():
    print("Loading model ...")
    model = load_model()

    combined_df = build_combined_df()
    g1020_df    = combined_df[combined_df["source"] == "G1020"].reset_index(drop=True)
    origa_df    = combined_df[combined_df["source"] == "ORIGA"].reset_index(drop=True)
    refuge_df   = combined_df[combined_df["source"] == "REFUGE"].reset_index(drop=True)

    print("\n=== Cross-Dataset Domain Shift Check ===")

    labels, probs = infer_dataset(model, g1020_df)
    report("G1020", labels, probs)

    labels, probs = infer_dataset(model, origa_df)
    report("ORIGA", labels, probs)

    labels, probs = infer_dataset(model, refuge_df)
    report("REFUGE train", labels, probs)

    print("\nDone. Gap < 15pp = acceptable calibration across cameras.")


if __name__ == "__main__":
    main()
