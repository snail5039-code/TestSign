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
        # Additive attention: score_t = v^T tanh(W h_t)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="artifacts/dataset.npz")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=96)   # 데이터 작으니 64~128 권장
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)

    # 프론트가 face=0이므로 학습도 face 영역을 0으로 맞춰서 분포 일치
    ap.add_argument("--zero_face", action="store_true", default=True)

    # 얼리스탑
    ap.add_argument("--patience", type=int, default=15)

    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"dataset npz not found: {args.npz}")

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"].astype(np.float32)  # (N,T,D)
    y = data["y"].astype(np.int64)

    classes = None
    for k in ["classes", "class_names", "labels"]:
        if k in data:
            classes = data[k]
            break
    if classes is None:
        classes = np.array([str(i) for i in range(int(y.max()) + 1)], dtype=object)
    classes = [str(c) for c in classes.tolist()]

    N, T, D = X.shape
    C = len(classes)
    print(f"[INFO] X={X.shape} y={y.shape} classes={C}")

    if args.zero_face:
        # pose(75) + face(210) + left(63) + right(63) = 411
        face_start = 75
        face_end = 75 + 210
        if D >= face_end:
            X[:, :, face_start:face_end] = 0.0
            print("[INFO] zero_face enabled (face slice -> 0)")
        else:
            print("[WARN] zero_face requested but D mismatch; skipped")

    # split (stratified)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=args.seed, stratify=y_tmp
    )

    # standardize features (fit on train only)
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

            tr_loss += loss.item()
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
            print(f"[E{epoch:03d}] loss={tr_loss:.4f} tr_acc={tr_acc:.4f} val_acc={va_acc:.4f} best={best_val:.4f} (pat={patience_left})")

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

    # save scaler
    scaler_path = os.path.join(args.out_dir, "attn_scaler.npz")
    np.savez(
        scaler_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )
    print(f"[SAVED] {scaler_path}")

    # save labels
    labels_path = os.path.join(args.out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {labels_path}")

    # save model config for server
    cfg_path = os.path.join(args.out_dir, "attn_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": int(D),
                "hidden": int(args.hidden),
                "layers": int(args.layers),
                "dropout": float(args.dropout),
                "zero_face": bool(args.zero_face),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[SAVED] {cfg_path}")


if __name__ == "__main__":
    main()
