# GlaucomaNet — Glaucoma Detection from Fundus Images

A binary glaucoma classifier trained on colour fundus photographs, benchmarked against
Hemelings et al. (2021) on the REFUGE validation set.

## Results

| Metric | GlaucomaNet (v4) | Hemelings et al. |
|---|---|---|
| **AUC-ROC** | **0.9158** | 0.8700 |
| Sensitivity | 85.0% | — |
| Specificity | 85.8% | — |
| Sensitivity @ 95% Specificity | 55.0% | — |
| Optimal threshold (Youden's J) | 0.385 | — |

**Anchor paper:** Hemelings et al., *Scientific Reports*, 2021 — [DOI: 10.1038/s41598-021-99605-1](https://doi.org/10.1038/s41598-021-99605-1)

---

## Architecture

| Component | Choice |
|---|---|
| Backbone | EfficientNet-B4 (timm, ImageNet pretrained) |
| Output | Single sigmoid neuron (binary) |
| Loss | Focal Loss (α=0.25, γ=2.0) |
| Optimiser | AdamW (lr=1e-4, weight_decay=1e-4) |
| LR schedule | CosineAnnealingLR |
| Fine-tuning | 2-phase: 5 warmup epochs (head only) + full fine-tune |
| Early stopping | Patience=10 on val AUC |
| Grad clipping | max_norm=1.0 |
| Input size | 384×384 + CLAHE (LAB L-channel) |
| Batch size | 16 |

---

## Datasets

| Dataset | Images | Glaucoma | Normal | Role |
|---|---|---|---|---|
| REFUGE train | 400 | 40 | 360 | Training |
| ORIGA | 650 | 168 | 482 | Training |
| G1020 | 1020 | 296 | 724 | Training |
| **Combined** | **2070** | **504** | **1566** | Training |
| REFUGE val | 400 | 40 | 360 | Validation / benchmark |
| REFUGE test | 400 | — | — | Held-out test |

A class-balanced `WeightedRandomSampler` ensures equal glaucoma/normal sampling per epoch regardless of dataset source.

---

## Project Structure

```
Glaucoma Project/
├── data_pipeline.py       # Dataset loading, CLAHE transforms, DataLoaders
├── train.py               # 2-phase EfficientNet-B4 training with Focal Loss
├── evaluate.py            # AUC, sensitivity, specificity, Youden's J threshold
├── gradcam.py             # Grad-CAM visualisations on REFUGE val TPs/FNs
├── domain_shift_check.py  # Cross-dataset calibration check (G1020, ORIGA)
├── app.py                 # Streamlit inference UI with Grad-CAM overlay
├── models/
│   └── best_model.pth     # Best checkpoint (epoch 19, AUC=0.9158)
└── outputs/
    ├── training_curves.png
    ├── roc_curve.png
    ├── confusion_matrix.png
    ├── gradcam_summary.png
    ├── gradcam/           # Individual TP/FN overlays
    └── metrics.json
```

---

## Setup

```bash
pip install torch torchvision timm pytorch-grad-cam streamlit pillow opencv-python scikit-learn matplotlib pandas
```

**Requirements:** Python 3.10+, CUDA GPU recommended (RTX 4060 used for development).

---

## Usage

### Training
```bash
python train.py
```
Trains for up to 30 epochs (5 warmup + 25 fine-tune) with early stopping. Saves best checkpoint to `models/best_model.pth`.

### Evaluation
```bash
python evaluate.py
```
Reports AUC, sensitivity, specificity, F1, and Sensitivity@95%Spec on REFUGE val. Saves ROC curve and confusion matrix to `outputs/`.

### Grad-CAM
```bash
python gradcam.py
```
Generates Grad-CAM overlays for the top-8 true positives and 4 false negatives from REFUGE val.

### Streamlit App
```bash
streamlit run app.py
```
Launches the inference UI. Upload a fundus image and click **Analyse** to get glaucoma probability + Grad-CAM overlay.

### Domain Shift Check
```bash
python domain_shift_check.py
```
Verifies the model generalises across camera types (G1020, ORIGA, REFUGE).

---

## Key Design Decisions

- **CLAHE preprocessing** — Contrast Limited Adaptive Histogram Equalization on the LAB L-channel normalises brightness/contrast differences across fundus cameras, reducing domain shift.
- **Focal Loss** — Down-weights easy negatives, forcing the model to focus on hard glaucoma cases and improving sensitivity.
- **2-phase fine-tuning** — Warming up only the classification head before unfreezing the backbone prevents early destruction of ImageNet representations.
- **Youden's J threshold** — Maximises sensitivity + specificity jointly rather than using a fixed 0.5 cutoff, better suited to clinical screening.

---

## Disclaimer

This tool is for **research and portfolio demonstration only**. It is not validated for clinical use and should not be used to make medical decisions.

---

*Developed as part of MTech research at IIT Hyderabad, in collaboration with L V Prasad Eye Institute (LVPEI).*
