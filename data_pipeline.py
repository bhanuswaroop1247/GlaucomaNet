import os
import json
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

BASE = r"C:\Projects\Glaucoma Project"
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

IMG_SIZE = 384

# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class CLAHE(object):
    """CLAHE on L channel (LAB). Normalises appearance across different fundus cameras."""
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr  = np.array(img, dtype=np.uint8)
        lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l    = self.clahe.apply(l)
        arr  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    CLAHE(clip_limit=2.0),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    CLAHE(clip_limit=2.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class FundusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["img_path"], row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ── Data loaders ──────────────────────────────────────────────────────────────
def build_combined_df():
    """Return combined training DataFrame with columns: filename, label, source, img_path."""
    rows = []

    # REFUGE train
    refuge_train_img = os.path.join(BASE, "REFUGE", "train", "Images")
    with open(os.path.join(BASE, "REFUGE", "train", "index.json")) as f:
        idx = json.load(f)
    for entry in idx.values():
        rows.append({
            "filename": entry["ImgName"],
            "label":    int(entry["Label"]),
            "source":   "REFUGE",
            "img_path": refuge_train_img,
        })

    # ORIGA
    origa_img = os.path.join(BASE, "ORIGA", "Images_Square")
    origa_df  = pd.read_csv(os.path.join(BASE, "ORIGA", "origa_info.csv"))
    for _, r in origa_df.iterrows():
        fname = r["Image"].split("/")[-1]
        if fname.startswith("."):
            continue
        rows.append({
            "filename": fname,
            "label":    int(r["Label"]),
            "source":   "ORIGA",
            "img_path": origa_img,
        })

    # G1020
    g1020_img = os.path.join(BASE, "G1020", "Images_Square")
    g1020_df  = pd.read_csv(os.path.join(BASE, "G1020", "G1020.csv"))
    for _, r in g1020_df.iterrows():
        rows.append({
            "filename": r["imageID"],
            "label":    int(r["binaryLabels"]),
            "source":   "G1020",
            "img_path": g1020_img,
        })

    return pd.DataFrame(rows)


def build_refuge_split(split):
    """Return DataFrame for REFUGE val or test split. Test set has no labels (label=-1)."""
    img_dir = os.path.join(BASE, "REFUGE", split, "Images")
    with open(os.path.join(BASE, "REFUGE", split, "index.json")) as f:
        idx = json.load(f)
    rows = [{"filename": e["ImgName"], "label": int(e.get("Label", -1)),
             "source": f"REFUGE_{split}", "img_path": img_dir}
            for e in idx.values()]
    return pd.DataFrame(rows)


def get_dataloaders(num_workers=0, batch_size=16):
    """Return (train_loader, val_loader, test_loader)."""
    combined_df = build_combined_df()

    label_counts     = combined_df["label"].value_counts()
    class_weight_map = {cls: 1.0 / cnt for cls, cnt in label_counts.items()}
    sample_weights   = combined_df["label"].map(class_weight_map).values
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(combined_df),
        replacement=True,
    )

    train_loader = DataLoader(
        FundusDataset(combined_df, transform=train_transform),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        FundusDataset(build_refuge_split("val"), transform=val_test_transform),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        FundusDataset(build_refuge_split("test"), transform=val_test_transform),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# ── Stats & visualisation ─────────────────────────────────────────────────────
def print_stats(combined_df, val_df, test_df):
    print("\n=== Class distribution per source ===")
    for src, grp in combined_df.groupby("source"):
        g = (grp["label"] == 1).sum()
        n = (grp["label"] == 0).sum()
        print(f"  {src:<10}  total={len(grp):4d}  glaucoma={g:3d}  normal={n:3d}")

    g_total = (combined_df["label"] == 1).sum()
    n_total = (combined_df["label"] == 0).sum()
    print(f"\n  Combined   total={len(combined_df):4d}  glaucoma={g_total:3d}  normal={n_total:3d}")
    print(f"\n  Train : {len(combined_df):4d} images")
    print(f"  Val   : {len(val_df):4d} images")
    print(f"  Test  : {len(test_df):4d} images")

    train_files = set(combined_df["filename"])
    val_files   = set(val_df["filename"])
    test_files  = set(test_df["filename"])
    print(f"\n  Train+Val  overlap : {len(train_files & val_files)} images (should be 0)")
    print(f"  Train+Test overlap : {len(train_files & test_files)} images (should be 0)")


def save_sample_grid(combined_df):
    glaucoma_rows = combined_df[combined_df["label"] == 1].sample(n=4, random_state=42)
    normal_rows   = combined_df[combined_df["label"] == 0].sample(n=4, random_state=42)
    sample_rows   = pd.concat([glaucoma_rows, normal_rows])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Sample Images — GlaucomaNet Training Set", fontsize=14)
    for ax, (_, row) in zip(axes.flatten(), sample_rows.iterrows()):
        img = Image.open(os.path.join(row["img_path"], row["filename"])).convert("RGB")
        img = img.resize((384, 384))
        ax.imshow(img)
        label_str = "Glaucoma" if row["label"] == 1 else "Normal"
        ax.set_title(f"{label_str}\n({row['source']})", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS, "sample_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Sample grid saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building data pipeline ...")
    combined_df = build_combined_df()
    val_df      = build_refuge_split("val")
    test_df     = build_refuge_split("test")
    get_dataloaders()
    print_stats(combined_df, val_df, test_df)
    save_sample_grid(combined_df)
    print("\nPhase 1 complete. DataLoaders ready.")
