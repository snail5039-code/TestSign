import os
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
SEED = 42
TARGET_FRAMES = 30          # JS에서 쓰던 FRAMES랑 맞춤
NUM_POSE = 33               # mediapipe pose landmarks
NUM_HAND = 21               # mediapipe hand landmarks
# face는 포맷이 제각각이라 일단 학습에서는 제외(원하면 쉽게 포함 가능)
USE_FACE = False

BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 256
LAYERS = 2
DROPOUT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold는 inference 쪽에서 confidence로 쓰는 게 안정적
DEFAULT_CONF_THRESHOLD = 0.70

# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

def dist2d(ax, ay, bx, by) -> float:
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx*dx + dy*dy)

def resample_list(seq: List[Any], target_len: int) -> List[Any]:
    """Uniform resample by index mapping. Works for list of frames."""
    if len(seq) == 0:
        return []
    if len(seq) == target_len:
        return seq
    out = []
    for i in range(target_len):
        idx = int(round(i * (len(seq) - 1) / (target_len - 1))) if target_len > 1 else 0
        out.append(seq[idx])
    return out

def safe_get(obj: Any, key: str, default=None):
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return default

def to_xyz_list(landmarks: Any) -> List[Optional[Dict[str, float]]]:
    """
    Normalize different landmark formats into list of dicts {x,y,z}.
    Input may be:
      - list of dicts with x/y(/z)
      - list of lists [x,y,z?]
      - dict with 'landmarks': [...]
    """
    if landmarks is None:
        return []
    if isinstance(landmarks, dict):
        if "landmarks" in landmarks:
            landmarks = landmarks["landmarks"]
        elif "points" in landmarks:
            landmarks = landmarks["points"]

    if not isinstance(landmarks, list):
        return []

    out = []
    for p in landmarks:
        if p is None:
            out.append(None)
            continue
        if isinstance(p, dict):
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            z = float(p.get("z", 0.0))
            out.append({"x": x, "y": y, "z": z})
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(p[0])
            y = float(p[1])
            z = float(p[2]) if len(p) >= 3 else 0.0
            out.append({"x": x, "y": y, "z": z})
        else:
            out.append(None)
    return out

def pick_primary_hand(frame: Dict[str, Any]) -> Tuple[List[Optional[Dict[str,float]]], str]:
    """
    Return (hand_landmarks, handedness_label)
    Handles many shapes:
      - frame["hands"] could be list of hands, each is list of 21 points
      - frame["hand"] single hand
      - handedness label might exist
    """
    handed = "Unknown"

    # common keys
    hands = safe_get(frame, "hands", None)
    hand = safe_get(frame, "hand", None)

    # handedness candidates
    handed = safe_get(frame, "handedness", handed)
    handed = safe_get(frame, "handednessLabel", handed)
    handed = safe_get(frame, "handedness_label", handed)

    if hand is not None:
        hlm = to_xyz_list(hand)
        return hlm, str(handed)

    if isinstance(hands, list) and len(hands) > 0:
        # if it's list of points directly (21 points)
        if len(hands) >= 21 and isinstance(hands[0], (dict, list, tuple)) and not isinstance(hands[0], list):
            # ambiguous; treat as single-hand list of points
            hlm = to_xyz_list(hands)
            return hlm, str(handed)

        # else hands = [hand0, hand1 ...]
        first = hands[0]
        hlm = to_xyz_list(first)
        return hlm, str(handed)

    return [], str(handed)

def get_pose_center_scale(pose_lm: List[Optional[Dict[str,float]]]) -> Tuple[float,float,float]:
    # shoulders 11,12 hips 23,24 (mediapipe pose)
    def get(i):
        if 0 <= i < len(pose_lm):
            return pose_lm[i]
        return None

    L_SH = get(11)
    R_SH = get(12)
    L_HIP = get(23)
    R_HIP = get(24)

    cx, cy, scale = 0.5, 0.5, 1.0
    if L_SH and R_SH:
        cx = (L_SH["x"] + R_SH["x"]) / 2.0
        cy = (L_SH["y"] + R_SH["y"]) / 2.0
        scale = dist2d(L_SH["x"], L_SH["y"], R_SH["x"], R_SH["y"])
    elif L_HIP and R_HIP:
        cx = (L_HIP["x"] + R_HIP["x"]) / 2.0
        cy = (L_HIP["y"] + R_HIP["y"]) / 2.0
        scale = dist2d(L_HIP["x"], L_HIP["y"], R_HIP["x"], R_HIP["y"])

    scale = max(scale, 1e-4)
    return cx, cy, scale

def normalize_landmarks(lms: List[Optional[Dict[str,float]]], cx: float, cy: float, scale: float, count: int) -> List[float]:
    out: List[float] = []
    # pad / trim
    lms = (lms + [None] * count)[:count]
    for p in lms:
        if not p:
            out.extend([0.0, 0.0, 0.0])
            continue
        nx = (float(p["x"]) - cx) / scale
        ny = (float(p["y"]) - cy) / scale
        nz = (float(p.get("z", 0.0))) / scale
        out.extend([clamp(nx, -5, 5), clamp(ny, -5, 5), clamp(nz, -5, 5)])
    return out

def build_feature(frame: Dict[str, Any]) -> np.ndarray:
    # pose
    pose_raw = safe_get(frame, "pose", None)
    pose_raw = safe_get(frame, "poseLandmarks", pose_raw)
    pose_raw = safe_get(frame, "pose_landmarks", pose_raw)
    pose_lm = to_xyz_list(pose_raw)

    has_pose = len(pose_lm) >= NUM_POSE
    if not has_pose:
        pose_lm = [None] * NUM_POSE

    # hand
    hand_lm, handed = pick_primary_hand(frame)
    if len(hand_lm) < NUM_HAND:
        hand_lm = (hand_lm + [None]*NUM_HAND)[:NUM_HAND]

    cx, cy, scale = get_pose_center_scale(pose_lm) if has_pose else (0.5, 0.5, 1.0)

    pose_vec = normalize_landmarks(pose_lm, cx, cy, scale, NUM_POSE)
    hand_vec = normalize_landmarks(hand_lm, cx, cy, scale, NUM_HAND)

    # handedness onehot
    if str(handed).lower().startswith("r"):
        onehot = [1.0, 0.0]
    elif str(handed).lower().startswith("l"):
        onehot = [0.0, 1.0]
    else:
        onehot = [0.0, 0.0]

    feat = np.array(pose_vec + hand_vec + onehot, dtype=np.float32)
    return feat

def parse_frames_from_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Return list of frame dicts.
    Tries common schemas:
      1) obj["frames"] is list of frame dicts
      2) obj["sequence"] is list
      3) obj itself is list -> treat as frames
      4) obj has pose/hands as lists over time -> convert to per-frame dicts
    """
    if isinstance(obj, dict):
        frames = safe_get(obj, "frames", None)
        if isinstance(frames, list) and len(frames) > 0:
            return [f if isinstance(f, dict) else {"pose": f} for f in frames]

        seq = safe_get(obj, "sequence", None)
        if isinstance(seq, list) and len(seq) > 0:
            return [f if isinstance(f, dict) else {"pose": f} for f in seq]

        # time-major keys case: pose/hands are list over time
        pose = safe_get(obj, "pose", None)
        hands = safe_get(obj, "hands", None)
        face = safe_get(obj, "face", None)

        if isinstance(pose, list) and len(pose) > 0 and (isinstance(pose[0], (dict, list))):
            # treat pose as frames
            T = len(pose)
            out = []
            for t in range(T):
                fr = {}
                fr["pose"] = pose[t]
                if isinstance(hands, list) and len(hands) == T:
                    fr["hands"] = hands[t]
                if isinstance(face, list) and len(face) == T:
                    fr["face"] = face[t]
                out.append(fr)
            return out

        # single frame dict?
        if any(k in obj for k in ["pose", "poseLandmarks", "hands", "hand", "face"]):
            return [obj]

    if isinstance(obj, list):
        # list of frames
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict):
            return obj
        # list of pose points -> treat as 1 frame
        return [{"pose": obj}]

    return []

# =========================
# Dataset
# =========================
@dataclass
class Sample:
    x: np.ndarray  # [T, D]
    y: int
    path: str

class JsonSequenceDataset(Dataset):
    def __init__(self, root: str, labels: List[str], files: List[Tuple[str,int]], augment: bool = False):
        self.root = root
        self.labels = labels
        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path, y = self.files[idx]
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        frames = parse_frames_from_json(obj)
        frames = resample_list(frames, TARGET_FRAMES) if len(frames) > 0 else [{} for _ in range(TARGET_FRAMES)]

        feats = [build_feature(fr) for fr in frames]
        x = np.stack(feats, axis=0)  # [T, D]

        # light augmentation (generalization)
        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), path

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # temporal jitter: small shift
        if random.random() < 0.35:
            shift = random.randint(-2, 2)
            if shift != 0:
                x = np.roll(x, shift, axis=0)

        # gaussian noise
        if random.random() < 0.50:
            noise = np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            x = x + noise

        # drop a few frames
        if random.random() < 0.25:
            k = random.randint(1, 3)
            for _ in range(k):
                t = random.randint(0, x.shape[0]-1)
                x[t, :] *= 0.0

        return x.astype(np.float32)

# =========================
# Model
# =========================
class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN*2),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN*2, num_classes),
        )

    def forward(self, x):
        # x: [B,T,D]
        out, _ = self.gru(x)           # [B,T,2H]
        last = out[:, -1, :]           # last timestep
        logits = self.head(last)
        return logits

# =========================
# Train / Eval
# =========================
def split_files_by_label(all_files: List[Tuple[str,int]], val_ratio: float = 0.2):
    # stratified split
    by_label: Dict[int, List[Tuple[str,int]]] = {}
    for p, y in all_files:
        by_label.setdefault(y, []).append((p, y))

    train, val = [], []
    for y, items in by_label.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = ce(logits, y)
        pred = logits.argmax(dim=-1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += float(loss.item()) * y.size(0)
    return loss_sum / max(1, total), correct / max(1, total)

def train(root: str, out_dir: str = "artifacts"):
    set_seed(SEED)

    # collect labels = subfolders
    labels = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if len(labels) == 0:
        raise RuntimeError(f"No label folders found in: {root}")

    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    all_files: List[Tuple[str,int]] = []
    for lab in labels:
        lab_dir = os.path.join(root, lab)
        for fn in sorted(os.listdir(lab_dir)):
            if fn.lower().endswith(".json"):
                all_files.append((os.path.join(lab_dir, fn), label_to_idx[lab]))

    if len(all_files) < 20:
        print(f"[WARN] total samples={len(all_files)} is small; training may overfit.")

    train_files, val_files = split_files_by_label(all_files, val_ratio=0.2)

    # peek input dim
    x0, _, p0 = JsonSequenceDataset(root, labels, [train_files[0]], augment=False)[0]
    input_dim = x0.shape[-1]

    print(f"labels({len(labels)}): {labels}")
    print(f"train={len(train_files)} val={len(val_files)}")
    print(f"input_dim={input_dim}, frames={TARGET_FRAMES}, device={DEVICE}")

    ds_train = JsonSequenceDataset(root, labels, train_files, augment=True)
    ds_val = JsonSequenceDataset(root, labels, val_files, augment=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)

    model = GRUClassifier(input_dim=input_dim, num_classes=len(labels)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        correct = 0

        for x, y, _ in dl_train:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = ce(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=-1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        va_loss, va_acc = evaluate(model, dl_val)

        print(f"[{epoch:03d}/{EPOCHS}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "labels": labels,
                    "input_dim": input_dim,
                    "frames": TARGET_FRAMES,
                    "conf_threshold": DEFAULT_CONF_THRESHOLD,
                },
                os.path.join(out_dir, "model_best.pt"),
            )

    print(f"Best val acc: {best_val_acc:.4f}")
    return os.path.join(out_dir, "model_best.pt")

def export_onnx(pt_path: str, out_dir: str = "artifacts"):
    ckpt = torch.load(pt_path, map_location="cpu")
    labels = ckpt["labels"]
    input_dim = ckpt["input_dim"]
    frames = ckpt["frames"]
    conf_threshold = ckpt.get("conf_threshold", DEFAULT_CONF_THRESHOLD)

    model = GRUClassifier(input_dim=input_dim, num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    # Save labels.json for JS
    labels_path = os.path.join(out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "frames": frames,
                "input_dim": input_dim,
                "conf_threshold": conf_threshold,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Export ONNX
    dummy = torch.zeros(1, frames, input_dim, dtype=torch.float32)
    onnx_path = os.path.join(out_dir, "model.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    print("Exported:", onnx_path)
    print("Saved labels:", labels_path)
    return onnx_path, labels_path

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="dataset root (folder contains label subfolders)")
    ap.add_argument("--out", type=str, default="artifacts", help="output dir")
    args = ap.parse_args()

    pt = train(args.data, out_dir=args.out)
    export_onnx(pt, out_dir=args.out)