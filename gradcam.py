import os
import sys
import numpy as np
import torch
import timm
from torch.amp import autocast
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, r"C:\Projects\Glaucoma Project")
from data_pipeline import build_refuge_split, val_test_transform

BASE     = r"C:\Projects\Glaucoma Project"
OUT      = os.path.join(BASE, "outputs")
GCAM_DIR = os.path.join(OUT, "gradcam")
CKPT     = os.path.join(BASE, "models", "best_model.pth")
DEVICE   = torch.device("cuda")

os.makedirs(GCAM_DIR, exist_ok=True)


def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    epoch = ckpt.get("epoch", "?")
    auc   = ckpt.get("val_auc") or ckpt.get("best_auc") or 0.0
    print(f"Loaded checkpoint: epoch {epoch}  val AUC={auc:.4f}")
    return model


@torch.no_grad()
def get_predictions(model, val_df):
    results = []
    for _, row in val_df.iterrows():
        img_path = os.path.join(row["img_path"], row["filename"])
        image    = Image.open(img_path).convert("RGB")
        tensor   = val_test_transform(image).unsqueeze(0).to(DEVICE)
        with autocast("cuda"):
            logit = model(tensor)
        prob_glaucoma = float(torch.sigmoid(logit).squeeze().cpu())
        pred_label    = int(prob_glaucoma >= 0.5)
        results.append({
            "filename":     row["filename"],
            "img_path":     row["img_path"],
            "true_label":   int(row["label"]),
            "pred_label":   pred_label,
            "prob_glaucoma": prob_glaucoma,
            "confidence":   max(prob_glaucoma, 1 - prob_glaucoma),
        })
    return results


def generate_gradcam_overlay(model, target_layer, img_path):
    image   = Image.open(img_path).convert("RGB").resize((384, 384))
    img_arr = np.array(image, dtype=np.float32) / 255.0
    tensor  = val_test_transform(image).unsqueeze(0).to(DEVICE)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]

    overlay = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
    return img_arr, overlay, grayscale_cam


def save_side_by_side(img_arr, overlay, meta, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img_arr); axes[0].set_title("Original", fontsize=10); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title("Grad-CAM", fontsize=10); axes[1].axis("off")

    true_str = "Glaucoma" if meta["true_label"] == 1 else "Normal"
    pred_str = "Glaucoma" if meta["pred_label"] == 1 else "Normal"
    status   = "TP" if (meta["true_label"] == 1 and meta["pred_label"] == 1) else "FN"
    fig.suptitle(
        f"{meta['filename']}  |  True: {true_str}  |  Pred: {pred_str}  "
        f"|  P(glaucoma)={meta['prob_glaucoma']:.2f}  [{status}]",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def save_summary_grid(tp_results, model, target_layer):
    n     = len(tp_results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for i, meta in enumerate(tp_results):
        img_path = os.path.join(meta["img_path"], meta["filename"])
        _, overlay, _ = generate_gradcam_overlay(model, target_layer, img_path)
        axes[i].imshow(overlay)
        axes[i].set_title(
            f"{meta['filename']}\nP={meta['prob_glaucoma']:.2f}",
            fontsize=7,
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Grad-CAM Summary — True Positive Glaucoma Cases (REFUGE Val)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUT, "gradcam_summary.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Summary grid saved: {path}")


def main():
    print("Loading model ...")
    model = load_model()

    # conv_head is the last 1x1 conv before global pool in timm EfficientNet-B4
    target_layer = model.conv_head

    print("Loading REFUGE val set ...")
    val_df = build_refuge_split("val")

    print("Running inference on all 400 val images ...")
    results = get_predictions(model, val_df)

    tp_results = [r for r in results if r["true_label"] == 1 and r["pred_label"] == 1]
    fn_results = [r for r in results if r["true_label"] == 1 and r["pred_label"] == 0]

    print(f"  Glaucoma positives in val: {sum(1 for r in results if r['true_label'] == 1)}")
    print(f"  True  Positives (TP): {len(tp_results)}")
    print(f"  False Negatives (FN): {len(fn_results)}")

    tp_selected = sorted(tp_results, key=lambda x: x["prob_glaucoma"], reverse=True)[:8]
    fn_selected = fn_results[:4]

    print(f"\nGenerating Grad-CAM for {len(tp_selected)} TPs and {len(fn_selected)} FNs ...")

    for meta in tp_selected + fn_selected:
        img_path  = os.path.join(meta["img_path"], meta["filename"])
        img_arr, overlay, _ = generate_gradcam_overlay(model, target_layer, img_path)
        tag       = "TP" if meta["pred_label"] == 1 else "FN"
        save_path = os.path.join(GCAM_DIR, f"{tag}_{meta['filename']}")
        save_side_by_side(img_arr, overlay, meta, save_path)

    print(f"  Individual overlays saved to: {GCAM_DIR}")

    print("\nBuilding TP summary grid ...")
    save_summary_grid(tp_selected, model, target_layer)

    print("\nChecking activation regions (centre = ONH/peripapillary) ...")
    centre_activations = 0
    for meta in tp_selected:
        img_path = os.path.join(meta["img_path"], meta["filename"])
        _, _, gcam = generate_gradcam_overlay(model, target_layer, img_path)
        h, w = gcam.shape
        centre_region = gcam[h//4: 3*h//4, w//4: 3*w//4]
        if centre_region.mean() > gcam.mean():
            centre_activations += 1

    pct = 100 * centre_activations / len(tp_selected) if tp_selected else 0
    print(f"  {centre_activations}/{len(tp_selected)} TPs ({pct:.0f}%) show primary activation "
          f"in centre region (ONH / peripapillary).")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
