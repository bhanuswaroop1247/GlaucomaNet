import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import timm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

sys.path.insert(0, r"C:\Projects\Glaucoma Project")
from data_pipeline import get_dataloaders

BASE   = r"C:\Projects\Glaucoma Project"
MODELS = os.path.join(BASE, "models");  os.makedirs(MODELS, exist_ok=True)
OUT    = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)

DEVICE        = torch.device("cuda")
EPOCHS        = 30
WARMUP_EPOCHS = 5       # frozen backbone, head-only warmup
LR_HEAD       = 1e-3
LR_FINETUNE   = 1e-4
WEIGHT_DECAY  = 1e-4
EARLY_STOP    = 10      # val-AUC patience
BEST_CKPT     = os.path.join(MODELS, "best_model.pth")


# ── Focal Loss (binary, single sigmoid output) ────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_f = targets.float()
        logits    = inputs.squeeze(1)
        bce       = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        probs     = torch.sigmoid(logits)
        p_t       = probs * targets_f + (1 - probs) * (1 - targets_f)
        alpha_t   = self.alpha * targets_f + (1 - self.alpha) * (1 - targets_f)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def build_model():
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=1)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    running_loss = 0.0
    criterion_eval = FocalLoss(alpha=0.25, gamma=2.0)
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast("cuda"):
            outputs = model(images)
            loss    = criterion_eval(outputs, labels)
        probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(loader.dataset)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def save_checkpoint(model, optimizer, epoch, auc):
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "val_auc":     auc,
        "best_auc":    auc,
    }, BEST_CKPT)


def plot_curves(train_losses, val_losses, val_aucs):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, "b-o", markersize=3)
    axes[0].set_title("Train Loss"); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, val_losses, "r-o", markersize=3)
    axes[1].set_title("Val Loss"); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, val_aucs, "g-o", markersize=3)
    axes[2].set_title("Val AUC"); axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(OUT, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved: {path}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading data ...")
    train_loader, val_loader, _ = get_dataloaders(num_workers=0, batch_size=16)
    print("Sampler: class-balanced (glaucoma = normal per epoch)")

    model     = build_model()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scaler    = GradScaler("cuda")

    best_auc   = 0.0
    best_epoch = 0
    train_losses, val_losses, val_aucs = [], [], []

    # ── Phase 1: warmup — freeze backbone, train head only ────────────────────
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 1 warmup — {WARMUP_EPOCHS} epochs, head only ({trainable:,} params)\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val AUC':>8}  {'LR':>8}")
    print("-" * 52)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=WARMUP_EPOCHS)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        t_loss        = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_auc = evaluate(model, val_loader)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_aucs.append(v_auc)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {t_loss:>10.4f}  {v_loss:>8.4f}  {v_auc:>8.4f}  {cur_lr:>8.6f}")

        if v_auc > best_auc:
            best_auc, best_epoch = v_auc, epoch
            save_checkpoint(model, optimizer, epoch, best_auc)
            print(f"         ** New best AUC={best_auc:.4f} — checkpoint saved **")

    # ── Phase 2: full fine-tune — unfreeze all layers ─────────────────────────
    for param in model.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    PHASE2 = EPOCHS - WARMUP_EPOCHS
    print(f"\nPhase 2 fine-tune — {PHASE2} epochs max, all layers ({total:,} params)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2)

    no_improve = 0
    for epoch in range(WARMUP_EPOCHS + 1, EPOCHS + 1):
        t_loss        = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        v_loss, v_auc = evaluate(model, val_loader)
        scheduler.step()

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_aucs.append(v_auc)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {t_loss:>10.4f}  {v_loss:>8.4f}  {v_auc:>8.4f}  {cur_lr:>8.6f}")

        if v_auc > best_auc:
            best_auc, best_epoch = v_auc, epoch
            save_checkpoint(model, optimizer, epoch, best_auc)
            print(f"         ** New best AUC={best_auc:.4f} — checkpoint saved **")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
                break

    print(f"\nBest val AUC : {best_auc:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint   : {BEST_CKPT}")
    plot_curves(train_losses, val_losses, val_aucs)


if __name__ == "__main__":
    main()
