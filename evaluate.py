import os
import sys
import json
import numpy as np
import torch
import timm
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    confusion_matrix, f1_score, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Projects\Glaucoma Project")
from data_pipeline import get_dataloaders

BASE   = r"C:\Projects\Glaucoma Project"
OUT    = os.path.join(BASE, "outputs")
CKPT   = os.path.join(BASE, "models", "best_model.pth")
DEVICE = torch.device("cuda")


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
def run_inference(model, val_loader):
    all_probs, all_labels = [], []
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        with autocast("cuda"):
            logits = model(images)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


def youden_threshold(labels, probs):
    """Optimal threshold via Youden's J (maximises sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def sensitivity_at_specificity(labels, probs, target_spec=0.95):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    spec = 1 - fpr
    idx  = np.argmin(np.abs(spec - target_spec))
    return float(tpr[idx]), float(thresholds[idx])


def compute_metrics(labels, probs):
    auc     = roc_auc_score(labels, probs)
    opt_thr = youden_threshold(labels, probs)

    preds = (probs >= opt_thr).astype(int)
    acc   = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(labels, preds, pos_label=1)
    sens_at_95spec, thresh_95 = sensitivity_at_specificity(labels, probs, 0.95)

    return {
        "auc_roc":               round(float(auc), 4),
        "optimal_threshold":     round(float(opt_thr), 4),
        "accuracy":              round(float(acc), 4),
        "sensitivity":           round(float(sens), 4),
        "specificity":           round(float(spec), 4),
        "f1_glaucoma":           round(float(f1), 4),
        "sensitivity_at_95spec": round(float(sens_at_95spec), 4),
        "threshold_at_95spec":   round(float(thresh_95), 4),
        "confusion_matrix":      cm.tolist(),
    }


def plot_roc(labels, probs, auc, opt_thr):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"GlaucomaNet  AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")
    ax.axvline(x=0.05, color="gray", linestyle=":", lw=1, label="Specificity = 95%")
    ax.plot(fpr[best_idx], tpr[best_idx], "ro", ms=8,
            label=f"Youden's J  thr={opt_thr:.3f}")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve — REFUGE Validation Set")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(OUT, "roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved: {path}")


def plot_confusion_matrix(labels, probs, opt_thr):
    preds = (probs >= opt_thr).astype(int)
    cm    = confusion_matrix(labels, preds, normalize="true")
    disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Glaucoma"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(f"Confusion Matrix (Normalized, thr={opt_thr:.3f}) — REFUGE Val")
    plt.tight_layout()
    path = os.path.join(OUT, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {path}")


def print_comparison(m):
    print("\n" + "=" * 60)
    print(f"  {'Metric':<30} {'Our Model':>10}  {'Hemelings et al.':>15}")
    print("=" * 60)
    print(f"  {'AUC-ROC':<30} {m['auc_roc']:>10.4f}  {'0.8700':>15}")
    print(f"  {'Optimal threshold (Youden J)':<30} {m['optimal_threshold']:>10.4f}  {'N/A':>15}")
    print(f"  {'Accuracy':<30} {m['accuracy']:>10.4f}  {'N/A':>15}")
    print(f"  {'Sensitivity':<30} {m['sensitivity']:>10.4f}  {'N/A':>15}")
    print(f"  {'Specificity':<30} {m['specificity']:>10.4f}  {'N/A':>15}")
    print(f"  {'F1 (Glaucoma)':<30} {m['f1_glaucoma']:>10.4f}  {'N/A':>15}")
    print(f"  {'Sensitivity @ 95% Spec':<30} {m['sensitivity_at_95spec']:>10.4f}  {'N/A':>15}")
    print("=" * 60)


def main():
    print("Loading model ...")
    model = load_model()

    print("Loading REFUGE val set ...")
    _, val_loader, _ = get_dataloaders(num_workers=0, batch_size=16)

    print("Running inference on 400 val images ...")
    labels, probs = run_inference(model, val_loader)

    print("Computing metrics ...")
    m = compute_metrics(labels, probs)

    print_comparison(m)

    plot_roc(labels, probs, m["auc_roc"], m["optimal_threshold"])
    plot_confusion_matrix(labels, probs, m["optimal_threshold"])

    metrics_path = os.path.join(OUT, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
