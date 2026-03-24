#!/usr/bin/env python3
"""
train_with_gemma.py -- Teacher-Student KD | Autonomous Self-Healing Training
=============================================================================
Fixes:
  [1] KD loss: batchmean / spatial_size (no warning, correct scaling)
  [2] LLM parser: handles list OR string from Gemini API
  [3] LLM parser: extract_json() safely strips ```json fences
  [4] CE_WEIGHTS override: rejects if not exactly NUM_CLASSES values
  [5] Gemini API: uses client.models.generate_content() (correct call)
  [6] Batch=16, LR=2e-4 scaled for RTX A4000 16GB
  [7] Teacher fully frozen
  [8] K-Means stratified val split (built once, fixed for entire run)
  [9] DYN config dict: LLM hot-swaps hyperparams live into optimizer
  [10] Plateau detection: PATIENCE_LIMIT=25 epochs -> auto LR heal
  [11] Hard exit at MAX_EPOCHS -- no infinite loop possible
  [12] Per-class IoU logged to console + email
  [13] Worst class injected into LLM prompt every call
  [14] Full DYN state + 8-epoch history sent to LLM for context
"""

import os
import json
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
import albumentations as A
from sklearn.cluster import KMeans
import resend
from google import genai as google_genai


# ------------------------------------------------------------------
# API CLIENTS
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
resend.api_key = RESEND_API_KEY
gemini_client = google_genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
print("[INIT] GEMINI_API_KEY : " + ("SET" if GEMINI_API_KEY else "MISSING"))
print("[INIT] RESEND_API_KEY : " + ("SET" if RESEND_API_KEY else "MISSING"))

EMAIL_FROM = "DroneTrainer <onboarding@resend.dev>"
EMAIL_TO = ["raghavk@duck.com"]


# ------------------------------------------------------------------
# DYNAMIC CONFIG -- every key here is hot-swappable by the LLM
# ------------------------------------------------------------------
DYN = {
    "LR":           2e-4,
    "KD_TEMP":      3.0,
    "ALPHA":        0.25,
    "BETA_CE":      0.30,
    "BETA_DICE":    0.30,
    "BETA_FOCAL":   0.10,
    "FEAT_KD_W":    0.05,
    "DROPOUT_P":    0.05,
    "FOCAL_GAMMA":  2.0,
    "LABEL_SMOOTH": 0.05,
}

# ------------------------------------------------------------------
# STATIC CONFIG
# ------------------------------------------------------------------
ZOOMED_IMG_DIR = "zoomed_out_1024/train/images"
ZOOMED_MSK_DIR = "zoomed_out_1024/train/masks"
ORIG_IMG_DIR   = "dataset_6_classes/train/images"
ORIG_MSK_DIR   = "dataset_6_classes/train/masks"

TEACHER_DIR      = "outputs/zoomed_teacher/best"
STUDENT_DIR      = "outputs/zoomed_student/best"
STUDENT_ARCH     = "nvidia/mit-b2"
OUT_DIR          = "outputs/mixed_student_v5"
RESUME_FILE      = "outputs/mixed_resume_v5.json"
LLM_HISTORY_FILE = "outputs/mixed_llm_history_v5.json"

NUM_CLASSES    = 4
CLASS_NAMES    = ["Water", "Road", "Built-up", "Background"]
CE_WEIGHTS     = [2.5, 3.0, 2.5, 0.5]
BATCH_SIZE     = 16
TARGET_MIOU    = 0.92
MAX_EPOCHS     = 300
PATIENCE_LIMIT = 25
LLM_EVERY      = 5
EMAIL_EVERY    = 5
EMA_DECAY      = 0.999
FEAT_PROJ_DIM  = 256
VAL_SPLIT      = 0.1
KMEANS_K       = 10
SEED           = 42
WARMUP_EPOCHS  = 2

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

cw_tensor = torch.tensor(CE_WEIGHTS, dtype=torch.float32, device=DEVICE)

# Global student ref so dropout override can reach the model
_student_ref = None


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def dyn_snapshot():
    return " | ".join(str(k) + "=" + str(v) for k, v in DYN.items())


# ------------------------------------------------------------------
# EMAIL
# ------------------------------------------------------------------
def send_email(subject, html):
    if not resend.api_key:
        print("[EMAIL] Skipped -- RESEND_API_KEY not set")
        return
    try:
        resend.Emails.send({"from": EMAIL_FROM, "to": EMAIL_TO,
                            "subject": subject, "html": html})
        print("[EMAIL] Sent: " + subject)
    except Exception as e:
        print("[EMAIL ERROR] " + str(e))


def epoch_email_html(epoch, t_miou, s_miou, s_ema_miou, losses,
                     best_miou, active_iou, llm_desc, patience):
    flag = "NEW BEST" if max(s_miou, s_ema_miou) >= best_miou else ""
    llm_blk = ("<p><b>LLM:</b> " + llm_desc + "</p>"
               if llm_desc else "<p><b>LLM:</b> No suggestion this epoch</p>")
    class_rows = "".join([
        "<tr><td>" + CLASS_NAMES[i] + "</td><td>" + str(round(float(active_iou[i]), 4)) + "</td></tr>"
        for i in range(NUM_CLASSES)
    ])
    worst = CLASS_NAMES[int(np.argmin(active_iou))]
    heal = ("<p style='color:red'><b>Plateau " + str(patience) + "/" + str(PATIENCE_LIMIT) + "</b></p>"
            if patience > 0 else "")
    return (
        "<h2>Epoch " + str(epoch) + " " + flag + "</h2>" +
        heal +
        "<table border='1' cellpadding='8' style='border-collapse:collapse'>" +
        "<tr><th>Model</th><th>mIoU</th><th>CE</th><th>Dice</th><th>Focal</th><th>KD</th><th>FeatKD</th></tr>" +
        "<tr><td>Teacher (Frozen)</td><td>" + str(round(t_miou, 4)) + "</td><td colspan='5'>n/a</td></tr>" +
        "<tr><td>Student</td><td>" + str(round(s_miou, 4)) + "</td>" +
        "<td>" + str(round(losses.get("ce", 0), 4)) + "</td>" +
        "<td>" + str(round(losses.get("dice", 0), 4)) + "</td>" +
        "<td>" + str(round(losses.get("focal", 0), 4)) + "</td>" +
        "<td>" + str(round(losses.get("kd", 0), 4)) + "</td>" +
        "<td>" + str(round(losses.get("feat", 0), 4)) + "</td></tr>" +
        "<tr><td>Student EMA</td><td>" + str(round(s_ema_miou, 4)) + "</td><td colspan='5'>n/a</td></tr>" +
        "</table><br>" +
        "<b>Per-Class IoU -- worst: " + worst + "</b>" +
        "<table border='1' cellpadding='6' style='border-collapse:collapse'>" +
        "<tr><th>Class</th><th>IoU</th></tr>" +
        class_rows +
        "</table>" +
        "<p>Best: <b>" + str(round(best_miou, 4)) + "</b> / Target: <b>" + str(TARGET_MIOU) + "</b><br>" +
        "<small>DYN: " + dyn_snapshot() + "</small><br>" +
        "<small>CE_WEIGHTS: " + str(CE_WEIGHTS) + "</small></p>" +
        llm_blk
    )


# ------------------------------------------------------------------
# EMA
# ------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay   = decay
        self.shadow  = {k: v.cpu().clone().float()
                        for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = (self.decay * self.shadow[k]
                              + (1 - self.decay) * v.cpu().float())

    @torch.no_grad()
    def apply(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        dev = next(model.parameters()).device
        model.load_state_dict(
            {k: v.to(dev) for k, v in self.shadow.items()}, strict=False)

    @torch.no_grad()
    def restore(self, model):
        if self._backup:
            model.load_state_dict(self._backup)
            self._backup = None

    def save(self, path):
        torch.save(self.shadow, path)

    @classmethod
    def load(cls, path, model, decay=EMA_DECAY):
        obj         = cls.__new__(cls)
        obj.decay   = decay
        obj.shadow  = torch.load(path, map_location="cpu")
        obj._backup = None
        return obj


# ------------------------------------------------------------------
# AUGMENTATION (aerial-safe)
# ------------------------------------------------------------------
TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                       rotate_limit=15, border_mode=0, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                             val_shift_limit=15, p=1.0),
    ], p=0.5),
    A.CLAHE(clip_limit=3.0, p=0.2),
    A.GaussNoise(var_limit=(10, 40), p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
                    fill_value=0, mask_fill_value=None, p=0.2),
])
VAL_AUG = A.Compose([])


# ------------------------------------------------------------------
# MASK REMAP
# ------------------------------------------------------------------
def remap_mask(mask_np):
    out = mask_np.astype(np.int64).copy()
    out[out > 3] = 3
    return out


# ------------------------------------------------------------------
# DATASET
# ------------------------------------------------------------------
class DroneDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, aug=None):
        self.pairs = pairs
        self.aug   = aug

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = np.load(img_path)
        msk = np.load(msk_path)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        msk = remap_mask(msk).astype(np.int64)
        if self.aug:
            out = self.aug(image=img, mask=msk.astype(np.uint8))
            img = out["image"]
            msk = out["mask"].astype(np.int64)
        return img, msk


def collate_fn(batch):
    imgs, msks = zip(*batch)
    return list(imgs), list(msks)


def _gather_pairs(img_dir, msk_dir):
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".npy"))
    return [(os.path.join(img_dir, f), os.path.join(msk_dir, f))
            for f in files if os.path.exists(os.path.join(msk_dir, f))]


# ------------------------------------------------------------------
# K-MEANS STRATIFIED SPLIT
# ------------------------------------------------------------------
def kmeans_stratified_split(all_pairs, val_split=0.1, k=10):
    print("[K-MEANS] Analyzing " + str(len(all_pairs)) + " masks...")
    features, valid_pairs = [], []
    for img_path, msk_path in all_pairs:
        try:
            msk    = remap_mask(np.load(msk_path))
            counts = np.bincount(msk.flatten(), minlength=NUM_CLASSES).astype(np.float32)
            features.append(counts / counts.sum())
            valid_pairs.append((img_path, msk_path))
        except Exception as e:
            print("  [SKIP] " + msk_path + ": " + str(e))
    features = np.array(features)
    k = min(k, len(valid_pairs) // 2)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(features)
    train_pairs, val_pairs = [], []
    for cid in range(k):
        idx = np.where(labels == cid)[0].tolist()
        random.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_pairs   += [valid_pairs[i] for i in idx[:n_val]]
        train_pairs += [valid_pairs[i] for i in idx[n_val:]]
    print("[K-MEANS] Split -> Train:" + str(len(train_pairs)) + " | Val:" + str(len(val_pairs)))
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        avg = features[idx].mean(axis=0)
        row = "  Cluster " + str(cid).zfill(2) + " (" + str(len(idx)).rjust(4) + " imgs) | "
        row += " ".join(CLASS_NAMES[c] + ":" + str(round(float(avg[c]), 2)) for c in range(NUM_CLASSES))
        print(row)
    return train_pairs, val_pairs


def get_fixed_dataloaders():
    all_pairs = (_gather_pairs(ZOOMED_IMG_DIR, ZOOMED_MSK_DIR)
                 + _gather_pairs(ORIG_IMG_DIR, ORIG_MSK_DIR))
    train_pairs, val_pairs = kmeans_stratified_split(all_pairs, VAL_SPLIT, KMEANS_K)
    random.shuffle(train_pairs)
    train_loader = DataLoader(
        DroneDataset(train_pairs, TRAIN_AUG),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        DroneDataset(val_pairs, VAL_AUG),
        batch_size=4, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=collate_fn)
    return train_loader, val_loader


# ------------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------------
def preprocess(proc, imgs, device):
    return proc(images=[Image.fromarray(img) for img in imgs],
                return_tensors="pt")["pixel_values"].to(device)


def labels_tensor(msks, device):
    return torch.from_numpy(
        np.stack([m if isinstance(m, np.ndarray) else np.array(m) for m in msks])
    ).long().to(device)


# ------------------------------------------------------------------
# TEACHER FORWARD (fully frozen, always no_grad)
# ------------------------------------------------------------------
@torch.no_grad()
def teacher_forward_4class(model_t, pixel_values):
    out         = model_t(pixel_values=pixel_values, output_hidden_states=True)
    class_probs = F.softmax(out.class_queries_logits, dim=-1)[..., :-1]
    mask_probs  = out.masks_queries_logits.sigmoid()
    sem6        = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    sem6        = F.interpolate(sem6, size=(512, 512),
                                mode="bilinear", align_corners=False)
    bg      = sem6[:, 3:4] + sem6[:, 4:5] + sem6[:, 5:6]
    logits4 = torch.cat([sem6[:, 0:1], sem6[:, 1:2], sem6[:, 2:3], bg], dim=1)
    t_feat  = out.encoder_hidden_states[-1] if out.encoder_hidden_states else None
    return logits4, t_feat


# ------------------------------------------------------------------
# LOSS FUNCTIONS
# ------------------------------------------------------------------
def ce_loss_fn(logits_up, gt):
    return F.cross_entropy(logits_up, gt, weight=cw_tensor,
                           label_smoothing=DYN["LABEL_SMOOTH"])


def dice_loss_fn(pred_softmax, target, smooth=1.0):
    loss = 0.0
    for c in range(NUM_CLASSES):
        p, t  = pred_softmax[:, c], (target == c).float()
        inter = (p * t).sum()
        loss += 1.0 - (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return loss / NUM_CLASSES


def focal_loss_fn(logits_up, target):
    ce = F.cross_entropy(logits_up, target, weight=cw_tensor, reduction="none")
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** DYN["FOCAL_GAMMA"] * ce).mean()


def kd_kl_loss(s_logits, t_logits):
    # FIX [1]: batchmean is mathematically correct for KL
    # Divide by spatial size to keep magnitude comparable to other losses
    T       = DYN["KD_TEMP"]
    s_up    = F.interpolate(s_logits, size=t_logits.shape[-2:],
                            mode="bilinear", align_corners=False)
    s_log   = F.log_softmax(s_up / T, dim=1)
    t_prob  = F.softmax(t_logits / T, dim=1)
    kd      = F.kl_div(s_log, t_prob, reduction="batchmean") * (T ** 2)
    spatial = t_logits.shape[-2] * t_logits.shape[-1]
    return kd / spatial


def feat_kd_loss_fn(s_feat, t_feat, proj):
    if s_feat is None or t_feat is None:
        return torch.tensor(0.0, device=DEVICE, requires_grad=True)
    t_feat = F.interpolate(t_feat.float(), size=s_feat.shape[-2:],
                           mode="bilinear", align_corners=False)
    t_proj = proj(t_feat)
    s_proj = s_feat[:, :FEAT_PROJ_DIM]
    if s_proj.shape[1] < FEAT_PROJ_DIM:
        pad    = torch.zeros(s_proj.shape[0],
                             FEAT_PROJ_DIM - s_proj.shape[1],
                             s_proj.shape[2], s_proj.shape[3], device=DEVICE)
        s_proj = torch.cat([s_proj, pad], dim=1)
    return F.mse_loss(s_proj, t_proj.detach())


def student_combined_loss(s_logits, t_logits4, gt, s_feat, t_feat, proj):
    s_up    = F.interpolate(s_logits, size=gt.shape[-2:],
                            mode="bilinear", align_corners=False)
    softmax = F.softmax(s_up, dim=1)
    l_ce    = ce_loss_fn(s_up, gt)
    l_dice  = dice_loss_fn(softmax, gt)
    l_focal = focal_loss_fn(s_up, gt)
    l_kd    = kd_kl_loss(s_logits, t_logits4)
    l_feat  = feat_kd_loss_fn(s_feat, t_feat, proj)
    total   = (DYN["ALPHA"]      * l_kd
               + DYN["BETA_CE"]   * l_ce
               + DYN["BETA_DICE"] * l_dice
               + DYN["BETA_FOCAL"] * l_focal
               + DYN["FEAT_KD_W"] * l_feat)
    return total, {
        "ce":    l_ce.item(),
        "dice":  l_dice.item(),
        "focal": l_focal.item(),
        "kd":    l_kd.item(),
        "feat":  l_feat.item(),
    }


# ------------------------------------------------------------------
# DROPOUT INJECTION
# ------------------------------------------------------------------
def inject_dropout(model_s):
    p = DYN["DROPOUT_P"]
    count = 0
    for _, m in model_s.named_modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            count += 1
    if hasattr(model_s, "decode_head") and hasattr(model_s.decode_head, "dropout"):
        model_s.decode_head.dropout.p = p
        count += 1
    print("  [DROPOUT] p=" + str(p) + " applied to " + str(count) + " layers")


# ------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------
@torch.no_grad()
def validate(model, proc, val_loader, is_teacher=False):
    model.eval()
    iou_sum = np.zeros(NUM_CLASSES)
    iou_cnt = np.zeros(NUM_CLASSES)
    for imgs, msks in val_loader:
        gt  = np.stack([m if isinstance(m, np.ndarray) else np.array(m) for m in msks])
        pix = preprocess(proc, imgs, DEVICE)
        with autocast(enabled=USE_AMP):
            if is_teacher:
                logits, _ = teacher_forward_4class(model, pix)
            else:
                logits = model(pixel_values=pix).logits
        logits = F.interpolate(logits.float(), size=gt.shape[-2:],
                               mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).cpu().numpy()
        for c in range(NUM_CLASSES):
            inter = np.logical_and(pred == c, gt == c).sum()
            uni   = np.logical_or(pred == c, gt == c).sum()
            if uni > 0:
                iou_sum[c] += inter / uni
                iou_cnt[c] += 1
    iou  = np.divide(iou_sum, np.maximum(iou_cnt, 1))
    miou = float(iou[iou_cnt > 0].mean())
    model.train()
    return miou, iou


# ------------------------------------------------------------------
# STATE
# ------------------------------------------------------------------
def load_resume():
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE) as f:
            s = json.load(f)
        print("  Resumed: epoch=" + str(s["epoch"]) + " best_s=" + str(round(s["best_miou"], 4)))
        return s
    return {"epoch": 0, "best_miou": 0.0, "best_t_miou": 0.0}


def save_resume(s):
    with open(RESUME_FILE, "w") as f:
        json.dump(s, f, indent=2)


def load_llm_history():
    if os.path.exists(LLM_HISTORY_FILE):
        with open(LLM_HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_llm_history(h):
    with open(LLM_HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)


# ------------------------------------------------------------------
# LLM SAFE PARSING -- FIX [2] + FIX [3]
# ------------------------------------------------------------------
def _normalize_llm_raw(raw):
    # FIX [2]: Gemini sometimes returns a list of parts instead of plain string
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        return "\n".join(str(x) for x in raw).strip()
    return str(raw).strip()


def _extract_json(text):
    # FIX [3]: Safely extract JSON from text that may contain ```json fences
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{") and chunk.endswith("}"):
                return chunk
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


# ------------------------------------------------------------------
# LLM OVERRIDE -- FIX [4] (CE_WEIGHTS length guard)
# ------------------------------------------------------------------
def apply_llm_override(proposal, opt_s):
    global cw_tensor
    param = str(proposal.get("p", "")).strip()
    raw   = str(proposal.get("v", "")).strip()
    desc  = str(proposal.get("d", ""))

    if param == "CE_WEIGHTS":
        try:
            new_w = json.loads(raw)
            if not isinstance(new_w, list):
                return "[OVERRIDE] Rejected CE_WEIGHTS -- not a list"
            if len(new_w) != NUM_CLASSES:
                return ("[OVERRIDE] Rejected CE_WEIGHTS -- need exactly " +
                        str(NUM_CLASSES) + " values, got " + str(len(new_w)))
            CE_WEIGHTS[:] = new_w
            cw_tensor = torch.tensor(CE_WEIGHTS, dtype=torch.float32, device=DEVICE)
            print("[OVERRIDE] CE_WEIGHTS -> " + str(new_w))
            return "CE_WEIGHTS updated to " + str(new_w)
        except Exception as e:
            return "[OVERRIDE] CE_WEIGHTS parse error: " + str(e)

    if param in DYN:
        try:
            new_val = float(raw)
            if new_val <= 0:
                return "[OVERRIDE] Rejected: " + param + "=" + str(new_val) + " (must be > 0)"
            old_val    = DYN[param]
            DYN[param] = new_val
            print("[OVERRIDE] " + param + ": " + str(old_val) + " -> " + str(new_val))

            if param == "LR" and opt_s is not None:
                for pg in opt_s.param_groups:
                    pg["lr"]         = new_val
                    pg["initial_lr"] = new_val
                print("[OVERRIDE] Optimizer LR updated to " + str(new_val))

            if param == "DROPOUT_P" and _student_ref is not None:
                inject_dropout(_student_ref)

            return param + ": " + str(round(old_val, 6)) + " -> " + str(round(new_val, 6)) + " | " + desc
        except Exception as e:
            return "[OVERRIDE] Failed " + param + ": " + str(e)

    return "[OVERRIDE] Unknown param '" + param + "' -- ignored"


# ------------------------------------------------------------------
# LLM AUTOTUNE -- FIX [5]: uses client.models.generate_content()
# ------------------------------------------------------------------
def run_autotune_gemini(epoch, miou, per_class_iou, patience, opt_s):
    if gemini_client is None:
        print("[LLM] Skipped -- no API key")
        return ""

    history  = load_llm_history()
    worst_c  = CLASS_NAMES[int(np.argmin(per_class_iou))]
    iou_str  = ", ".join(CLASS_NAMES[c] + "=" + str(round(float(per_class_iou[c]), 3))
                         for c in range(NUM_CLASSES))
    hist_lines = [
        "ep" + str(h["epoch"]) + ":mIoU=" + str(h["miou"]) +
        ",s=" + str(h["strategy"]) +
        ",p=" + str(h.get("param", "")) +
        ",v=" + str(h.get("value", "")) +
        ",status=" + str(h["status"])
        for h in history[-8:]
    ]
    last_ok = "none"
    for h in reversed(history):
        if h.get("status") == "APPLIED":
            last_ok = str(h.get("param", "")) + "=" + str(h.get("value", "")) + " @ ep" + str(h["epoch"])
            break

    dyn_str = json.dumps(DYN)
    ce_str  = json.dumps(CE_WEIGHTS)
    prompt  = (
        "You are an autonomous semantic segmentation training agent with authority "
        "to directly modify hyperparameters.\n"
        "RULES: 1) Teacher is frozen - do not suggest teacher changes. "
        "2) Do not repeat last action. "
        "3) If plateau_epochs>=" + str(PATIENCE_LIMIT) + ", make a large strategy change. "
        "4) CE_WEIGHTS must be a JSON array of exactly " + str(NUM_CLASSES) + " values. "
        "5) Return ONLY valid JSON, no markdown, no explanation.\n"
        "STATUS: epoch=" + str(epoch) + ", best_mIoU=" + str(round(miou, 4)) +
        ", target=" + str(TARGET_MIOU) +
        ", plateau=" + str(patience) + "/" + str(PATIENCE_LIMIT) + "\n"
        "PER-CLASS IoU: " + iou_str + "\n"
        "WORST CLASS: " + worst_c + "\n"
        "CURRENT DYN: " + dyn_str + "\n"
        "CE_WEIGHTS: " + ce_str + "\n"
        "HISTORY (last 8): " + ("; ".join(hist_lines) if hist_lines else "none") + "\n"
        "LAST SUCCESSFUL APPLY: " + last_ok + "\n"
        "MODIFIABLE: any key in DYN or CE_WEIGHTS (JSON array)\n"
        "Give one improvement targeting [" + worst_c + "], return exactly:\n"
        '{"s":1,"d":"<suggestion in English>","p":"<param_name>","v":"<new_value_string>"}\n'
        "Strategy codes: 1=general hyperparams 2=backbone 3=loss weights 4=featKD 5=augment 6=LR"
    )

    try:
        print("[LLM] Querying Gemini ep=" + str(epoch) +
              " mIoU=" + str(round(miou, 4)) +
              " worst=" + worst_c +
              " patience=" + str(patience) + "...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIX [5]: correct Gemini SDK call
            out = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
        # .text is always a plain string -- no list/lstrip issues
        raw_text = _normalize_llm_raw(out.text)
        print("[LLM] Raw: " + raw_text[:400])

    except Exception as e:
        print("[LLM] API error: " + str(e))
        history.append({
            "epoch":       epoch,
            "miou":        round(miou, 4),
            "strategy":    None,
            "description": str(e),
            "param":       "",
            "value":       "",
            "status":      "ERROR",
            "time":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_llm_history(history)
        return ""

    clean = _extract_json(raw_text)

    try:
        proposal = json.loads(clean)
    except json.JSONDecodeError as e:
        print("[LLM] JSON parse error: " + str(e) + " | clean=" + clean[:120])
        history.append({
            "epoch":       epoch,
            "miou":        round(miou, 4),
            "strategy":    "?",
            "description": "parse_error: " + clean[:80],
            "param":       "",
            "value":       "",
            "status":      "PARSE_ERROR",
            "time":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        save_llm_history(history)
        return "[PARSE_ERROR]"

    s_names = {1: "Hyperparams", 2: "Backbone", 3: "Loss",
               4: "FeatKD",      5: "Augment",  6: "LR"}
    s     = proposal.get("s", "?")
    desc  = proposal.get("d", "")
    param = proposal.get("p", "")
    val   = proposal.get("v", "")

    apply_result = apply_llm_override(proposal, opt_s)
    print("[LLM] Strategy=" + str(s) + " (" + str(s_names.get(s, "?")) + ") -> " + apply_result)

    history.append({
        "epoch":        epoch,
        "miou":         round(miou, 4),
        "strategy":     s,
        "description":  desc,
        "param":        param,
        "value":        val,
        "apply_result": apply_result,
        "status":       "APPLIED",
        "time":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    save_llm_history(history)
    return "[" + str(s_names.get(s, "?")) + "] " + apply_result


# ------------------------------------------------------------------
# MODEL LOADERS
# ------------------------------------------------------------------
def load_teacher_model():
    best_t = os.path.join(OUT_DIR, "best_teacher")
    src    = best_t if os.path.isdir(best_t) else TEACHER_DIR
    print("  Teacher: " + src)
    proc  = Mask2FormerImageProcessor.from_pretrained(src, do_reduce_labels=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        src, num_labels=6, ignore_mismatched_sizes=True).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  [TEACHER] Fully frozen.")
    return proc, model


def load_student_model(from_epoch=-1):
    if from_epoch > 0:
        path = os.path.join(OUT_DIR, "epoch-" + str(from_epoch).zfill(3))
        if os.path.isdir(path):
            print("  Student: " + path)
            proc  = SegformerImageProcessor.from_pretrained(path, do_reduce_labels=False)
            model = SegformerForSemanticSegmentation.from_pretrained(
                path, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(DEVICE)
            inject_dropout(model)
            return proc, model
    src = STUDENT_DIR if os.path.isdir(STUDENT_DIR) else STUDENT_ARCH
    print("  Student: " + src)
    proc  = SegformerImageProcessor.from_pretrained(src, do_reduce_labels=False)
    model = SegformerForSemanticSegmentation.from_pretrained(
        src, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(DEVICE)
    inject_dropout(model)
    return proc, model


def load_feat_proj(from_epoch=-1):
    proj = nn.Conv2d(1024, FEAT_PROJ_DIM, kernel_size=1).to(DEVICE)
    if from_epoch > 0:
        path = os.path.join(OUT_DIR, "epoch-" + str(from_epoch).zfill(3), "feat_proj.pt")
        if os.path.exists(path):
            proj.load_state_dict(torch.load(path, map_location=DEVICE))
            print("  feat_proj loaded from epoch " + str(from_epoch))
    return proj


def build_optimizer(model_s, feat_proj, start_epoch):
    opt = AdamW([
        {"params": model_s.segformer.parameters(),   "lr": DYN["LR"] * 0.1},
        {"params": model_s.decode_head.parameters(), "lr": DYN["LR"]},
        {"params": feat_proj.parameters(),           "lr": DYN["LR"]},
    ], weight_decay=0.01)
    for pg in opt.param_groups:
        pg["initial_lr"] = pg["lr"]
    sched = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=1, eta_min=1e-6)
    for _ in range(start_epoch - 1):
        sched.step()
    return opt, sched


# ------------------------------------------------------------------
# TRAIN ONE EPOCH
# ------------------------------------------------------------------
def train_student(model_s, model_t, proc_s, proc_t,
                  loader, opt, proj, ema, scaler, epoch):
    model_s.train()
    log = {"ce": [], "dice": [], "focal": [], "kd": [], "feat": []}

    for step, (imgs, msks) in enumerate(loader, 1):
        gt    = labels_tensor(msks, DEVICE)
        s_pix = preprocess(proc_s, imgs, DEVICE)
        t_pix = preprocess(proc_t, imgs, DEVICE)

        with autocast(enabled=USE_AMP):
            t_logits4, t_feat = teacher_forward_4class(model_t, t_pix)
            s_out    = model_s(pixel_values=s_pix, output_hidden_states=True)
            s_logits = s_out.logits
            s_feat   = s_out.hidden_states[-1] if s_out.hidden_states else None
            total, ldict = student_combined_loss(
                s_logits, t_logits4, gt, s_feat, t_feat, proj)

        opt.zero_grad()
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model_s)

        for k, v in ldict.items():
            log[k].append(v)

        if step % 20 == 0:
            print("  ep" + str(epoch) + " step" + str(step).rjust(4) +
                  " | ce=" + str(round(ldict["ce"], 3)) +
                  " dice=" + str(round(ldict["dice"], 3)) +
                  " focal=" + str(round(ldict["focal"], 3)) +
                  " kd=" + str(round(ldict["kd"], 4)) +
                  " feat=" + str(round(ldict["feat"], 3)) +
                  " lr=" + "{:.2e}".format(opt.param_groups[-1]["lr"]))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in log.items()}


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    global _student_ref

    print("Device:" + str(DEVICE) + "  AMP:" + str(USE_AMP) +
          "  Batch:" + str(BATCH_SIZE) + "  LR:" + "{:.0e}".format(DYN["LR"]))
    print("Target:" + str(TARGET_MIOU) + "  MAX_EPOCHS:" + str(MAX_EPOCHS) +
          "  PATIENCE:" + str(PATIENCE_LIMIT))

    state     = load_resume()
    epoch     = state["epoch"] + 1
    best_miou = state["best_miou"]
    best_t    = state.get("best_t_miou", 0.0)

    if best_miou >= TARGET_MIOU:
        print("Target already achieved (" + str(best_miou) + "). Exiting.")
        return

    proc_t, model_t = load_teacher_model()
    proc_s, model_s = load_student_model(from_epoch=epoch - 1)
    feat_proj       = load_feat_proj(from_epoch=epoch - 1)
    _student_ref    = model_s

    model_s.train()
    opt_s, sched_s = build_optimizer(model_s, feat_proj, start_epoch=epoch)
    scaler = GradScaler(enabled=USE_AMP)

    ema_path = os.path.join(OUT_DIR, "ema_shadow.pt")
    if os.path.exists(ema_path):
        ema = EMA.load(ema_path, model_s)
        print("  EMA loaded from disk.")
    else:
        ema = EMA(model_s)

    train_loader, val_loader = get_fixed_dataloaders()

    if best_t == 0.0:
        print("[INIT] Evaluating frozen teacher baseline...")
        best_t, t_iou = validate(model_t, proc_t, val_loader, is_teacher=True)
        for c in range(NUM_CLASSES):
            print("  Teacher " + CLASS_NAMES[c].ljust(12) + ": " + str(round(float(t_iou[c]), 4)))
        print("  Frozen Teacher mIoU: " + str(round(best_t, 4)))
        state["best_t_miou"] = best_t
        save_resume(state)

    send_email(
        "train_with_gemma.py started -- ep" + str(epoch) + " target=" + str(TARGET_MIOU),
        "<h2>train_with_gemma.py launched</h2>"
        "<p>Device:" + str(DEVICE) + " | Batch:" + str(BATCH_SIZE) + " | LR:" + "{:.0e}".format(DYN["LR"]) + "</p>"
        "<p>Frozen Teacher mIoU: <b>" + str(round(best_t, 4)) + "</b></p>"
        "<p>Student best: <b>" + str(round(best_miou, 4)) + "</b> / Target: <b>" + str(TARGET_MIOU) + "</b></p>"
        "<p>PATIENCE_LIMIT: " + str(PATIENCE_LIMIT) + "</p>",
    )

    patience_counter = 0

    # ------------------------------------------------------------------
    # MAIN LOOP -- guaranteed to exit: MAX_EPOCHS hard cap
    # ------------------------------------------------------------------
    while best_miou < TARGET_MIOU and epoch <= MAX_EPOCHS:
        print("\n" + "=" * 72)
        print("EPOCH " + str(epoch) + "/" + str(MAX_EPOCHS) +
              "  best=" + str(round(best_miou, 4)) +
              "  teacher=" + str(round(best_t, 4)) +
              "  patience=" + str(patience_counter) + "/" + str(PATIENCE_LIMIT))
        print("DYN: " + dyn_snapshot())
        print("=" * 72)

        if epoch <= WARMUP_EPOCHS:
            wf = epoch / WARMUP_EPOCHS
            for pg in opt_s.param_groups:
                pg["lr"] = pg["initial_lr"] * wf

        t0       = time.time()
        s_losses = train_student(model_s, model_t, proc_s, proc_t,
                                 train_loader, opt_s, feat_proj, ema, scaler, epoch)

        if epoch > WARMUP_EPOCHS:
            sched_s.step()

        elapsed = time.time() - t0

        s_miou, s_iou         = validate(model_s, proc_s, val_loader)
        ema.apply(model_s)
        s_ema_miou, s_ema_iou = validate(model_s, proc_s, val_loader)
        ema.restore(model_s)

        best_this  = max(s_miou, s_ema_miou)
        active_iou = s_ema_iou if s_ema_miou >= s_miou else s_iou
        worst_idx  = int(np.argmin(active_iou))

        print("\n  elapsed  : " + str(int(elapsed)) + "s")
        print("  Teacher  : " + str(round(best_t, 4)) + " (frozen)")
        print("  Student  : " + str(round(s_miou, 4)) +
              "  EMA:" + str(round(s_ema_miou, 4)) +
              "  Best:" + str(round(best_miou, 4)))
        for c in range(NUM_CLASSES):
            tag = " <- worst" if c == worst_idx else ""
            print("    " + CLASS_NAMES[c].ljust(12) + ": " + str(round(float(active_iou[c]), 4)) + tag)

        ep_path = os.path.join(OUT_DIR, "epoch-" + str(epoch).zfill(3))
        os.makedirs(ep_path, exist_ok=True)
        model_s.save_pretrained(ep_path)
        proc_s.save_pretrained(ep_path)
        torch.save(feat_proj.state_dict(), os.path.join(ep_path, "feat_proj.pt"))
        ema.save(ema_path)

        new_best = False
        if best_this > best_miou:
            best_miou        = best_this
            new_best         = True
            patience_counter = 0
            bp = os.path.join(OUT_DIR, "best")
            if s_ema_miou >= s_miou:
                ema.apply(model_s)
                model_s.save_pretrained(bp)
                proc_s.save_pretrained(bp)
                ema.restore(model_s)
                print("  New best mIoU=" + str(round(best_miou, 4)) + " (EMA)")
            else:
                model_s.save_pretrained(bp)
                proc_s.save_pretrained(bp)
                print("  New best mIoU=" + str(round(best_miou, 4)) + " (student)")
        else:
            patience_counter += 1
            print("  [PATIENCE] " + str(patience_counter) + "/" + str(PATIENCE_LIMIT))

        if patience_counter >= PATIENCE_LIMIT:
            new_lr = max(DYN["LR"] * 0.3, 1e-6)
            print("[HEAL] Plateau! LR " + "{:.2e}".format(DYN["LR"]) + " -> " + "{:.2e}".format(new_lr))
            DYN["LR"] = new_lr
            for pg in opt_s.param_groups:
                pg["lr"]         = new_lr
                pg["initial_lr"] = new_lr
            patience_counter = 0
            send_email(
                "Plateau Healed -- Epoch " + str(epoch),
                "<h2>Auto-Heal Triggered</h2>"
                "<p>No improvement for " + str(PATIENCE_LIMIT) + " epochs.</p>"
                "<p>LR reduced to <b>" + "{:.2e}".format(new_lr) + "</b></p>"
                "<p>Best mIoU: <b>" + str(round(best_miou, 4)) + "</b></p>",
            )

        state.update({"epoch": epoch, "best_miou": best_miou, "best_t_miou": best_t})
        save_resume(state)

        llm_desc = ""
        if epoch % LLM_EVERY == 0:
            llm_desc = run_autotune_gemini(
                epoch, best_miou, active_iou, patience_counter, opt_s)

        if epoch % EMAIL_EVERY == 0 or new_best:
            subj = ("New Best " + str(round(best_miou, 4)) + " -- Epoch " + str(epoch)
                    if new_best
                    else "Epoch " + str(epoch) + " | mIoU=" + str(round(best_this, 4)) +
                         " | p=" + str(patience_counter) + "/" + str(PATIENCE_LIMIT))
            send_email(subj, epoch_email_html(
                epoch, best_t, s_miou, s_ema_miou, s_losses,
                best_miou, active_iou, llm_desc, patience_counter))

        if best_miou >= TARGET_MIOU:
            print("\n" + "=" * 72)
            print("  TARGET REACHED: " + str(round(best_miou, 4)) + " >= " + str(TARGET_MIOU))
            print("=" * 72)
            break

        epoch += 1

    reason = ("TARGET REACHED" if best_miou >= TARGET_MIOU
              else "MAX_EPOCHS (" + str(MAX_EPOCHS) + ") exhausted")
    send_email(
        "Training Complete -- " + reason + " -- Best=" + str(round(best_miou, 4)),
        "<h2>Training finished</h2>"
        "<p>Reason: <b>" + reason + "</b></p>"
        "<p>Best Student mIoU: <b>" + str(round(best_miou, 4)) + "</b></p>"
        "<p>Frozen Teacher mIoU: <b>" + str(round(best_t, 4)) + "</b></p>"
        "<p>Checkpoint: " + OUT_DIR + "/best</p>",
    )
    print("Done. Best=" + str(round(best_miou, 4)) + " -> " + OUT_DIR + "/best")


if __name__ == "__main__":
    main()
