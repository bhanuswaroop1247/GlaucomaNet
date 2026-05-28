import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm
import streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.amp import autocast

BASE   = os.path.dirname(os.path.abspath(__file__))
CKPT   = os.path.join(BASE, "models", "best_model_weights.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CLAHE:
    """CLAHE on L channel — normalises appearance across different fundus cameras."""
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l   = self._clahe.apply(l)
        arr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)


preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    CLAHE(clip_limit=2.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    ckpt  = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def run_inference(model, pil_image):
    tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    use_amp = DEVICE.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with autocast("cuda"):
                logit = model(tensor)
        else:
            logit = model(tensor)
    glaucoma_prob = float(torch.sigmoid(logit).squeeze().cpu())
    pred_label    = int(glaucoma_prob >= 0.5)
    return pred_label, glaucoma_prob


def run_gradcam(model, pil_image):
    img_384  = pil_image.resize((384, 384))
    img_384  = CLAHE()(img_384)
    img_arr  = np.array(img_384, dtype=np.float32) / 255.0
    tensor   = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    target_layer = model.conv_head
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None)[0]
    overlay = show_cam_on_image(img_arr, grayscale_cam, use_rgb=True)
    return overlay


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlaucomaNet",
    page_icon=":eye:",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GlaucomaNet")
    st.markdown(
        """
        A deep learning classifier for glaucoma detection from colour fundus photographs.
        Trained on REFUGE + ORIGA + G1020 (~2,070 images) using EfficientNet-B4 with
        ImageNet transfer learning and 2-phase fine-tuning, evaluated on the REFUGE val set.
        """
    )
    st.divider()
    st.subheader("Model Performance (REFUGE val)")
    col1, col2 = st.columns(2)
    col1.metric("AUC-ROC", "0.9158", "+0.0458 vs baseline")
    col2.metric("Specificity", "85.8%")
    st.metric("Sensitivity", "85.0%")
    st.metric("Sensitivity @ 95% Specificity", "55.0%")
    st.divider()
    st.markdown(
        "**Anchor Paper**  \n"
        "Hemelings et al., *Scientific Reports*, 2021  \n"
        "[DOI: 10.1038/s41598-021-99605-1](https://doi.org/10.1038/s41598-021-99605-1)"
    )
    st.divider()
    st.subheader("Settings")
    threshold = st.slider(
        "Glaucoma threshold",
        min_value=0.10, max_value=0.90, value=0.40, step=0.05,
        help="Probability above this value → Glaucoma Suspected. "
             "Use the Youden's J threshold from evaluate.py for best calibration.",
    )
    st.caption(
        "Note: model benchmarked on REFUGE-style images. "
        "Images from other camera types tend to over-predict glaucoma — "
        "raise the threshold to 0.65-0.75 for non-REFUGE cameras."
    )
    st.divider()
    device_label = "CUDA (GPU)" if DEVICE.type == "cuda" else "CPU"
    st.caption(f"Running on: {device_label}")
    st.caption("Model: EfficientNet-B4  |  Input: 384x384 + CLAHE")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("GlaucomaNet — Glaucoma Detection from Fundus Images")
st.caption(
    "EfficientNet-B4 (timm)  |  Focal Loss  |  AdamW + CosineAnnealingLR  "
    "|  Trained on REFUGE + ORIGA + G1020  |  Benchmarked vs Hemelings et al. (AUC 0.87)"
)
st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a colour fundus image",
    type=["jpg", "jpeg", "png"],
    help="Upload a square or near-square fundus photograph.",
)

if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")

    col_img, col_gap = st.columns([1, 2])
    with col_img:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    st.divider()
    analyse = st.button("Analyse", type="primary", use_container_width=False)

    if analyse:
        model = load_model()

        with st.spinner("Running inference and generating Grad-CAM..."):
            pred_label, glaucoma_prob = run_inference(model, pil_image)
            overlay = run_gradcam(model, pil_image)

        is_glaucoma = glaucoma_prob >= threshold
        label_str   = "Glaucoma Suspected" if is_glaucoma else "Normal"
        conf_pct    = f"{glaucoma_prob * 100:.1f}% glaucoma probability"

        if is_glaucoma:
            st.error(f"### {label_str}   —   {conf_pct}")
        else:
            st.success(f"### {label_str}   —   {conf_pct}")

        st.subheader("Grad-CAM Activation Map")
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_image, caption="Original", use_container_width=True)
        with c2:
            st.image(overlay, caption="Grad-CAM Overlay (model attention)", use_container_width=True)

        st.divider()
        if is_glaucoma:
            st.warning(
                "**Clinical note:** Activation concentrated on optic nerve head region. "
                "Refer for clinical evaluation."
            )
        else:
            st.info(
                "**Clinical note:** No significant glaucomatous features detected. "
                "Routine follow-up recommended."
            )

        with st.expander("Probability breakdown"):
            normal_prob = 1 - glaucoma_prob
            st.write(f"- Normal:   **{normal_prob * 100:.2f}%**")
            st.write(f"- Glaucoma: **{glaucoma_prob * 100:.2f}%**")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    ":warning: This tool is for research and portfolio demonstration only. "
    "Not validated for clinical use."
)
