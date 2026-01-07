# scripts/train_attn_lstm.py
import os
import json
import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def batch_acc(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


def load_label_map(label_map_path: str):
    """
    labels_ko.json 예시:
    {
      "display": {
        "yes": "네",
        "no": "아니요"
      }
    }
    """
    if not label_map_path:
        return {}
    if not os.path.exists(label_map_path):
        print(f"[WARN] label_map not found: {label_map_path} (skip)")
        return {}
    with open(label_map_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    display = obj.get("display", obj.get("map", obj.get("labels", {})))
    if not isinstance(display, dict):
        return {}
    return {str(k): str(v) for k, v in display.items()}


def save_scaler_json(out_dir: str, mean: np.ndarray, scale: np.ndarray):
    path = os.path.join(out_dir, "attn_scaler.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"mean": mean.astype(float).tolist(), "scale": scale.astype(float).tolist()},
            f,
            ensure_ascii=False,
        )
    print(f"[SAVED] {path}")


def export_onnx_legacy(model: nn.Module, onnx_path: str, T: int, D: int, opset: int = 18):
    """
    ✅ 레거시 exporter 강제: dynamo=False
    AttnBiLSTM에서 흔히 터지는 onnxscript 경로를 회피.
    """
    model.eval()
    dummy = torch.zeros(1, T, D, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # ✅ 핵심
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/dataset.npz")
    ap.add_argument("--out_dir", default="artifacts")

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)

    # 프론트가 face=0이면 학습도 face 영역을 0으로 맞추는 게 유리
    ap.add_argument("--zero_face", action="store_true", default=True)
    ap.add_argument("--no_zero_face", action="store_true", default=False)

    ap.add_argument("--patience", type=int, default=15)

    # export
    ap.add_argument("--export_onnx", action="store_true", default=False)
    ap.add_argument("--opset", type=int, default=18)

    # ✅ 한국어 표시 매핑 파일
    ap.add_argument("--label_map", type=str, default="", help="labels_ko.json path (optional)")

    args = ap.parse_args()

    if args.no_zero_face:
        args.zero_face = False

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"dataset npz not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float32)  # (N,T,D)
    y = data["y"].astype(np.int64)

    # class names
    classes = None
    for k in ["class_names", "classes", "labels"]:
        if k in data:
            classes = data[k]
            break
    if classes is None:
        classes = np.array([str(i) for i in range(int(y.max()) + 1)], dtype=object)
    classes = [str(c) for c in classes.tolist()]

    N, T, D = X.shape
    C = len(classes)
    print(f"[INFO] X={X.shape} y={y.shape} classes={C}")
    print(f"[INFO] out_dir={args.out_dir}")

    # === counts (build_dataset.py가 저장) ===
    left_cnt  = int(data.get("leftHand_count", 21))
    right_cnt = int(data.get("rightHand_count", 21))
    pose_cnt  = int(data.get("pose_count", 25))
    face_cnt  = int(data.get("face_count", 70))

    left_dim  = left_cnt * 3
    right_dim = right_cnt * 3
    pose_dim  = pose_cnt * 3
    face_dim  = face_cnt * 3

    face_start = left_dim + right_dim + pose_dim
    face_end = face_start + face_dim

    if args.zero_face and face_dim > 0 and D >= face_end:
        X[:, :, face_start:face_end] = 0.0
        print(f"[INFO] zero_face enabled: slice [{face_start}:{face_end}] -> 0")
    else:
        print(f"[INFO] zero_face={args.zero_face} (skipped or not applicable) D={D}, face_dim={face_dim}")

    # split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )

    # standardize (fit train only)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, D))

    def transform(Z):
        return scaler.transform(Z.reshape(-1, D)).reshape(-1, T, D).astype(np.float32)

    X_train_s = transform(X_train)
    X_val_s = transform(X_val)
    X_test_s = transform(X_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    model = AttnBiLSTMClassifier(
        input_dim=D,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_classes=C,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(SeqDataset(X_train_s, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val_s, y_val), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test_s, y_test), batch_size=args.batch, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = 0
    patience_left = args.patience

    model_path = os.path.join(args.out_dir, "attn_lstm.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, nb = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tr_loss += float(loss.item())
            tr_acc += batch_acc(logits, yb)
            nb += 1

        tr_loss /= max(nb, 1)
        tr_acc /= max(nb, 1)

        model.eval()
        va_acc, vb = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_acc += batch_acc(logits, yb)
                vb += 1
        va_acc /= max(vb, 1)

        improved = va_acc > best_val
        if improved:
            best_val = va_acc
            best_epoch = epoch
            patience_left = args.patience
            torch.save(model.state_dict(), model_path)
        else:
            patience_left -= 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[E{epoch:03d}] loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
                f"val_acc={va_acc:.4f} best={best_val:.4f} (pat={patience_left})"
            )

        if patience_left <= 0:
            print(f"[EARLY STOP] best_epoch={best_epoch} best_val={best_val:.4f}")
            break

    # test
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    te_acc, tb = 0.0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            te_acc += batch_acc(logits, yb)
            tb += 1
    te_acc /= max(tb, 1)

    print(f"[DONE] best_val={best_val:.4f} test_acc={te_acc:.4f}")
    print(f"[SAVED] {model_path}")

    # save scaler (npz + json)
    scaler_npz = os.path.join(args.out_dir, "attn_scaler.npz")
    np.savez(
        scaler_npz,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    print(f"[SAVED] {scaler_npz}")
    save_scaler_json(args.out_dir, scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32))

    # save labels (+ korean display map if provided)
    display_map = load_label_map(args.label_map)

    # classes에 존재하는 키만 남기기(프론트 안정성)
    display = {}
    for c in classes:
        if c in display_map:
            display[c] = display_map[c]

    labels_path = os.path.join(args.out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": classes,
                "display": display,     # ✅ { "yes": "네", ... }
                "T": int(T),
                "D": int(D),
                "model": {
                    "type": "AttnBiLSTM",
                    "hidden": int(args.hidden),
                    "layers": int(args.layers),
                    "dropout": float(args.dropout),
                },
                "preprocess": {
                    "standardize": True,
                    "zero_face": bool(args.zero_face),
                    "modality_order": ["leftHand", "rightHand", "pose", "face"],
                    "counts": {
                        "leftHand": int(left_cnt),
                        "rightHand": int(right_cnt),
                        "pose": int(pose_cnt),
                        "face": int(face_cnt),
                    },
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SAVED] {labels_path}")

    # export ONNX
    if args.export_onnx:
        cpu_model = AttnBiLSTMClassifier(
            input_dim=D,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            num_classes=C,
            dropout=args.dropout,
        ).cpu()
        cpu_model.load_state_dict(torch.load(model_path, map_location="cpu"))

        onnx_path = os.path.join(args.out_dir, "attn_lstm.onnx")
        export_onnx_legacy(cpu_model, onnx_path, T=T, D=D, opset=args.opset)
        print(f"[SAVED] {onnx_path}")


if __name__ == "__main__":
    main()
