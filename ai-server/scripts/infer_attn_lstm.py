# scripts/infer_attn_lstm.py
import json
import numpy as np
import torch
import torch.nn as nn

# 너의 build_dataset.py가 만든 feature(X)랑 "동일한 D"의 입력이 필요함.
# 여기서는 "이미 (T,D) feature 시퀀스를 만들었다"는 전제 예시.
# 실제로는 프론트에서 쌓은 (T,D)를 그대로 가져오거나,
# 서버에서 동일한 feature-building을 해야 함.

class AttnBiLSTMClassifier(nn.Module):
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
        h, _ = self.lstm(x)
        scores = self.attn_v(torch.tanh(self.attn_W(h))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = torch.sum(alpha * h, dim=1)
        return self.head(ctx)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def main():
    ART = "artifacts"
    cfg = json.load(open(f"{ART}/attn_config.json", "r", encoding="utf-8"))
    labels = json.load(open(f"{ART}/labels.json", "r", encoding="utf-8"))
    scaler = np.load(f"{ART}/attn_scaler.npz")

    T = cfg["T"]
    D = cfg["input_dim"]
    zero_face = cfg["zero_face"]
    counts = cfg["counts"]
    classes = labels["classes"]
    labelKoMap = labels.get("labelKoMap", {}) or {}

    # 모델 로드
    ckpt = torch.load(f"{ART}/attn_lstm.pt", map_location="cpu")
    model = AttnBiLSTMClassifier(
        input_dim=D,
        hidden_dim=ckpt["hidden"],
        num_layers=ckpt["layers"],
        num_classes=len(classes),
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 예시 입력: (T,D) feature 시퀀스 (여긴 임시)
    X = np.zeros((T, D), dtype=np.float32)

    # zero_face 적용(학습과 동일하게)
    if zero_face:
        left_dim = counts["leftHand"] * 3
        right_dim = counts["rightHand"] * 3
        pose_dim = counts["pose"] * 3
        face_dim = counts["face"] * 3
        face_start = left_dim + right_dim + pose_dim
        face_end = face_start + face_dim
        if D >= face_end and face_dim > 0:
            X[:, face_start:face_end] = 0.0

    # scaler 적용(학습과 동일)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    Xs = (X - mean) / (scale + 1e-9)

    # 모델 추론
    with torch.no_grad():
        xb = torch.from_numpy(Xs[None, :, :]).float()  # (1,T,D)
        logits = model(xb).numpy()[0]
        prob = softmax(logits)

    top = int(np.argmax(prob))
    en_label = classes[top]
    ko_label = (labelKoMap.get(en_label, "") or "").strip()
    shown = ko_label if ko_label else en_label

    print("pred_en:", en_label)
    print("pred_ko:", ko_label)
    print("shown  :", shown)
    print("conf   :", float(prob[top]))

if __name__ == "__main__":
    main()
