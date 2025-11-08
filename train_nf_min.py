# train_nf_min.py — balanced FocalLoss with alpha smoothing, macro-F1 early stop, NF recall floor

import os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, recall_score

# ===== Config =====
DATA_DIR = "data"
IMG_SIZE = 224
BASE_BATCH = 16                    # On CPU this will be halved below
EPOCHS = 50
PATIENCE = 7
LR = 1e-4
WEIGHT_DECAY = 1e-4
OUTDIR = "runs_nf_min"             # unified output dir
USE_PRETRAINED = False
WARMUP_HEAD_EPOCHS = 3
SAVE_METRIC = "macro_f1"           # 'macro_f1' or 'nf_recall'
NF_RECALL_FLOOR = 0.60             # only save if NF recall >= this
# Focal settings
GAMMA = 1.5
ALPHA_NF_BOOST = 1.0               # no extra boost now
ALPHA_SMOOTH_LAMBDA = 0.30         # 0~1, blend with uniform to reduce extremeness
# Resume settings
RESUME_FROM_BEST = True            # load runs_nf_min/best.pt if exists

os.makedirs(OUTDIR, exist_ok=True)

# ----- Device -----
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
BATCH = BASE_BATCH if device != "cpu" else max(4, BASE_BATCH // 2)
NUM_WORKERS = 0
PIN_MEMORY = (device == "cuda")
print(f"[INFO] device={device} batch={BATCH}")

# ----- Seed -----
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ----- Transforms -----
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ----- Datasets / Loaders -----
train_ds = datasets.ImageFolder(Path(DATA_DIR)/"train", transform=train_tf)
val_ds   = datasets.ImageFolder(Path(DATA_DIR)/"val",   transform=val_tf)
assert set(train_ds.classes) == set(val_ds.classes), "Class sets in train/ and val/ must match"
num_classes = len(train_ds.classes)
class_to_idx = train_ds.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
print("Classes:", train_ds.classes)
print("class_to_idx:", class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# ----- Per-class alpha (inverse-frequency) + smoothing -----
counts = np.zeros(num_classes, dtype=np.int64)
for _, y in train_ds.samples:
    counts[y] += 1
print("Train counts:", {idx_to_class[i]: int(c) for i,c in enumerate(counts)})

inv = 1.0 / np.maximum(counts, 1)
alpha_vec = inv / inv.sum()

# optional per-class manual boost (here none)
nf_idx = class_to_idx.get("NF", 0)
alpha_vec[nf_idx] *= ALPHA_NF_BOOST
alpha_vec = alpha_vec / alpha_vec.sum()

# smooth towards uniform
uniform = np.ones_like(alpha_vec) / len(alpha_vec)
alpha_vec = (1 - ALPHA_SMOOTH_LAMBDA) * alpha_vec + ALPHA_SMOOTH_LAMBDA * uniform

alpha_torch = torch.tensor(alpha_vec.astype(np.float32),
                           device=torch.device("mps") if device=="mps" else device)
print("alpha (smoothed):", {idx_to_class[i]: float(alpha_vec[i]) for i in range(num_classes)})

# ----- Model -----
import timm
model = timm.create_model("efficientnet_b0", pretrained=USE_PRETRAINED, num_classes=num_classes)
model = model.to(torch.device("mps") if device=="mps" else device)

# Warmup: freeze backbone for first few epochs
for name, p in model.named_parameters():
    if "classifier" not in name:
        p.requires_grad = True if WARMUP_HEAD_EPOCHS<=0 else False

# Resume from best (weights only)
best_ckpt = os.path.join(OUTDIR, "best.pt")
if RESUME_FROM_BEST and os.path.exists(best_ckpt):
    try:
        ckpt = torch.load(best_ckpt, map_location=torch.device("mps") if device=="mps" else device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INFO] Resumed weights from {best_ckpt}")
    except Exception as e:
        print("[WARN] Failed to resume:", e)

# ----- Loss / Optim / Sched -----
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [N]
        probs = torch.softmax(logits, dim=1)
        pt = probs[torch.arange(len(target), device=logits.device), target].clamp_min(1e-6)
        alpha_w = 1.0 if self.alpha is None else self.alpha[target]
        loss = alpha_w * (1 - pt) ** self.gamma * ce
        return loss.mean()

criterion = FocalLoss(gamma=GAMMA, alpha=alpha_torch)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

def to_dev(x):
    if device == "cuda": return x.cuda(non_blocking=True)
    if device == "mps":  return x.to(torch.device("mps"))
    return x

@torch.no_grad()
def evaluate(return_details=False):
    """Evaluate on val set; return loss, macroF1, macroAUC, nf_recall (and optionally y_true/y_pred)."""
    model.eval()
    ys, yhat, prob_mat = [], [], []
    loss_sum, n = 0.0, 0
    for x, y in val_loader:
        x, y = to_dev(x), to_dev(y)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(1)
        ys.extend(y.detach().cpu().tolist())
        yhat.extend(pred.detach().cpu().tolist())
        prob_mat.append(probs.detach().cpu())
        n += x.size(0)

    y_true = np.array(ys, dtype=int)
    y_pred = np.array(yhat, dtype=int)
    prob_mat = torch.cat(prob_mat).numpy() if len(prob_mat)>0 else None

    macro_f1 = f1_score(y_true, y_pred, average="macro") if len(y_true)>0 else float('nan')
    nf_recall = recall_score((y_true==nf_idx), (y_pred==nf_idx)) if len(y_true)>0 else float('nan')

    try:
        y_true_oh = np.eye(num_classes)[y_true]
        auc_macro = roc_auc_score(y_true_oh, prob_mat, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = float('nan')

    avg_loss = loss_sum / max(n, 1)
    if return_details:
        return avg_loss, macro_f1, auc_macro, nf_recall, y_true, y_pred
    return avg_loss, macro_f1, auc_macro, nf_recall

best_score, bad = -1.0, 0

for epoch in range(1, EPOCHS+1):
    # Unfreeze backbone after warmup
    if epoch == WARMUP_HEAD_EPOCHS + 1:
        for p in model.parameters():
            p.requires_grad = True
        print(f"[INFO] Unfreeze backbone at epoch {epoch}")

    model.train()
    t0 = time.time()
    run_loss, seen = 0.0, 0

    for i, (x, y) in enumerate(train_loader, 1):
        x, y = to_dev(x), to_dev(y)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        run_loss += loss.item() * bs
        seen += bs
        if i % 10 == 0:
            print(f"  batch {i}/{len(train_loader)}  loss={loss.item():.4f}")

    val_loss, val_f1, val_auc, nf_recall = evaluate()
    score = val_f1 if SAVE_METRIC == "macro_f1" else nf_recall
    scheduler.step(score)

    print(f"Epoch {epoch:02d} | train_loss={(run_loss/max(seen,1)):.4f} | "
          f"val_loss={val_loss:.4f} | macroF1={val_f1:.4f} | NF_recall={nf_recall:.4f} | "
          f"macroAUC={val_auc:.4f} | time={time.time()-t0:.1f}s")

    # Save only if NF recall is acceptable
    if (np.isnan(nf_recall) or nf_recall >= NF_RECALL_FLOOR) and score > best_score:
        best_score, bad = score, 0
        torch.save({"model": model.state_dict(), "classes": train_ds.classes}, best_ckpt)
        print("  -> saved:", best_ckpt)
    else:
        bad += 1
        if bad >= PATIENCE:
            print("Early stopping."); break

print("Best score:", best_score, " ckpt:", best_ckpt)

# Final detailed evaluation
ckpt = torch.load(best_ckpt, map_location=torch.device("mps") if device=="mps" else device)
model.load_state_dict(ckpt["model"])
val_loss, val_f1, val_auc, nf_recall, y_true, y_pred = evaluate(return_details=True)
print(f"\n== Final VAL == loss={val_loss:.4f} | macroF1={val_f1:.4f} | NF_recall={nf_recall:.4f} | macroAUC={val_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=train_ds.classes, digits=4))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_true, y_pred))
