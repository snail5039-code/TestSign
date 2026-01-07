import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ----------------------------
# Model (must match training)
# ----------------------------
class AttnBiLSTMClassifier(nn.Module):
    """
    BiLSTM -> attention pooling over time -> classifier
    x: (B, T, D)
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        h, _ = self.lstm(x)  # (B, T, 2H)
        scores = self.attn_v(torch.tanh(self.attn_W(h))).squeeze(-1)  # (B, T)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)            # (B, T, 1)
        ctx = torch.sum(alpha * h, dim=1)                             # (B, 2H)
        logits = self.head(ctx)                                       # (B, C)
        return logits


# ----------------------------
# Helpers
# ----------------------------
def _safe_list(x):
    return x if isinstance(x, list) else []


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_X_from_sample(sample_obj: dict, T: int, D: int, counts: dict, order: list, zero_face: bool):
    """
    Expects JSON like:
    {
      "label": "hello",
      "label_ko": "안녕하세요",   # optional
      "frames": [
        {"pose":[...], "face":[...], "leftHand":[...], "rightHand":[...]},
        ...
      ]
    }
    Each modality is a flat list length = count*3
    """
    frames = sample_obj.get("frames", [])
    frames = frames if isinstance(frames, list) else []

    # pad/trim frames to T (hold-last)
    if len(frames) >= T:
        frames = frames[:T]
    else:
        last = frames[-1] if frames else {}
        frames = frames + [last] * (T - len(frames))

    # modality dims
    def dim(mod):  # count * 3
        return int(counts.get(mod, 0)) * 3

    # offsets for optional zero_face
    # order is something like ["leftHand","rightHand","pose","face"]
    offsets = {}
    cur = 0
    for m in order:
        offsets[m] = (cur, cur + dim(m))
        cur += dim(m)

    X = np.zeros((T, D), dtype=np.float32)

    for t, fr in enumerate(frames):
        fr = fr if isinstance(fr, dict) else {}
        feats = []

        for m in order:
            arr = fr.get(m, [])
            arr = _safe_list(arr)

            target_len = dim(m)
            if target_len <= 0:
                continue

            # pad/trim
            if len(arr) >= target_len:
                vec = np.array(arr[:target_len], dtype=np.float32)
            else:
                vec = np.zeros((target_len,), dtype=np.float32)
                if len(arr) > 0:
                    vec[:len(arr)] = np.array(arr, dtype=np.float32)

            feats.append(vec)

        if feats:
            v = np.concatenate(feats, axis=0)
        else:
            v = np.zeros((D,), dtype=np.float32)

        # final pad/trim to D
        if v.shape[0] >= D:
            v = v[:D]
        else:
            v = np.pad(v, (0, D - v.shape[0]))

        # apply zero_face if configured
        if zero_face and "face" in offsets:
            s, e = offsets["face"]
            if 0 <= s < e <= D:
                v[s:e] = 0.0

        X[t] = v

    return X  # (T, D)


def softmax_np(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dataset root (contains label folders)")
    ap.add_argument("--artifacts", required=True, help="artifacts dir (attn_lstm.pt, attn_scaler.npz, attn_config.json, labels.json)")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = all files, else cap per class")
    ap.add_argument("--threshold", type=float, default=0.0, help="if >0, report abstain when top1_prob < threshold")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    art = Path(args.artifacts)

    cfg = load_json(str(art / "attn_config.json"))
    T = int(cfg["T"])
    D = int(cfg["input_dim"])
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 2))
    dropout = float(cfg.get("dropout", 0.35))
    zero_face = bool(cfg.get("zero_face", False))
    order = cfg.get("modality_order", ["leftHand", "rightHand", "pose", "face"])
    counts = cfg.get("counts", {"leftHand": 21, "rightHand": 21, "pose": 25, "face": 70})

    labels_obj = load_json(str(art / "labels.json"))
    classes = labels_obj.get("classes") or labels_obj.get("labels")
    if classes is None:
        raise RuntimeError("labels.json must contain {classes:[...]} or {labels:[...]}")

    classes = [str(x) for x in classes]
    C = len(classes)
    lab2idx = {lab: i for i, lab in enumerate(classes)}

    sc = np.load(str(art / "attn_scaler.npz"))
    mean = sc["mean"].astype(np.float32)
    scale = sc["scale"].astype(np.float32)
    if mean.shape[0] != D or scale.shape[0] != D:
        raise RuntimeError(f"Scaler dim mismatch: mean/scale={mean.shape}/{scale.shape} vs D={D}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttnBiLSTMClassifier(D, hidden, layers, C, dropout=dropout).to(device)
    sd = torch.load(str(art / "attn_lstm.pt"), map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # gather files per class folder
    files_by_class = {}
    for d in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        lab = d.name
        fps = sorted(d.glob("*.json"))
        if args.max_per_class and args.max_per_class > 0:
            fps = fps[: args.max_per_class]
        if fps:
            files_by_class[lab] = fps

    # stats
    total = 0
    correct = 0
    abstain = 0
    confusion = defaultdict(lambda: defaultdict(int))
    worst = []  # (prob, gt, pred, path)

    with torch.no_grad():
        for gt_lab, fps in files_by_class.items():
            if gt_lab not in lab2idx:
                # dataset folder name not in labels.json
                continue

            for fp in fps:
                obj = load_json(str(fp))
                gt = str(obj.get("label", gt_lab))

                X = build_X_from_sample(obj, T=T, D=D, counts=counts, order=order, zero_face=zero_face)
                # standardize
                X = ((X - mean) / (scale + 1e-12)).astype(np.float32)

                xb = torch.from_numpy(X).unsqueeze(0).to(device)  # (1,T,D)
                logits = model(xb).squeeze(0).detach().cpu().numpy()  # (C,)
                probs = softmax_np(logits)

                pred_idx = int(np.argmax(probs))
                pred_lab = classes[pred_idx]
                top1 = float(probs[pred_idx])

                total += 1

                if args.threshold and top1 < args.threshold:
                    abstain += 1
                    confusion[gt]["<ABSTAIN>"] += 1
                    worst.append((top1, gt, "<ABSTAIN>", str(fp)))
                    continue

                confusion[gt][pred_lab] += 1
                if pred_lab == gt:
                    correct += 1
                else:
                    worst.append((top1, gt, pred_lab, str(fp)))

    acc = correct / max(1, total)
    print(f"[SUMMARY] total={total} correct={correct} acc={acc:.4f} abstain={abstain} threshold={args.threshold}")

    # per-class accuracy
    print("\n[PER-CLASS]")
    for gt in sorted(confusion.keys()):
        row = confusion[gt]
        n = sum(row.values())
        ok = row.get(gt, 0)
        print(f"  {gt:>16s}: {ok:4d}/{n:4d}  acc={ok/max(1,n):.3f}")

    # top confusions (off-diagonal)
    pairs = []
    for gt, row in confusion.items():
        for pred, cnt in row.items():
            if pred != gt and cnt > 0:
                pairs.append((cnt, gt, pred))
    pairs.sort(reverse=True)
    print("\n[TOP CONFUSIONS]")
    for cnt, gt, pred in pairs[:15]:
        print(f"  {gt} -> {pred}: {cnt}")

    # worst examples
    worst.sort(key=lambda x: x[0])  # low confidence first
    print("\n[WORST EXAMPLES] (lowest top1 prob among wrong/abstain)")
    for p, gt, pred, path in worst[:20]:
        print(f"  p={p:.4f}  gt={gt}  pred={pred}  file={path}")


if __name__ == "__main__":
    main()
