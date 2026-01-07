# scripts/infer_one_attn.py
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn


# ---------------- Model (training과 동일) ----------------
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


# ---------------- Utils ----------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resample_frames(frames, target_T: int):
    """Uniform index resample (JS/파이프라인과 유사)."""
    if not frames:
        return [{} for _ in range(target_T)]
    if len(frames) == target_T:
        return frames
    out = []
    n = len(frames)
    for i in range(target_T):
        idx = int(round(i * (n - 1) / (target_T - 1))) if target_T > 1 else 0
        out.append(frames[idx])
    return out

def as_float_list(v, expected_len: int):
    """
    v가 list면 float로 변환 + 길이 맞춤.
    없거나 이상하면 0으로 채움.
    """
    if not isinstance(v, list):
        return [0.0] * expected_len
    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    if len(out) >= expected_len:
        return out[:expected_len]
    return out + [0.0] * (expected_len - len(out))

def build_X_from_collect_json(obj, T: int, D: int, counts: dict):
    """
    Collect.jsx 출력(JSON) 기준:
    obj.frames[t] 안에 pose/face/leftHand/rightHand가 flat array로 들어있음.
    순서는 modality_order = ["leftHand","rightHand","pose","face"] 로 고정.
    """
    frames = obj.get("frames", None)
    if not isinstance(frames, list):
        # 혹시 frames가 없는 변형 포맷이면 단일 프레임 취급
        frames = [obj]

    frames = resample_frames(frames, T)

    left_dim  = int(counts.get("leftHand", 21)) * 3
    right_dim = int(counts.get("rightHand", 21)) * 3
    pose_dim  = int(counts.get("pose", 25)) * 3
    face_dim  = int(counts.get("face", 70)) * 3

    X = np.zeros((T, D), dtype=np.float32)

    for t in range(T):
        fr = frames[t] if isinstance(frames[t], dict) else {}
        left  = as_float_list(fr.get("leftHand"), left_dim)
        right = as_float_list(fr.get("rightHand"), right_dim)
        pose  = as_float_list(fr.get("pose"), pose_dim)
        face  = as_float_list(fr.get("face"), face_dim)

        vec = left + right + pose + face

        # D가 더 크거나 작을 수 있으니 안전하게 맞춤
        if len(vec) >= D:
            X[t] = np.array(vec[:D], dtype=np.float32)
        else:
            tmp = vec + [0.0] * (D - len(vec))
            X[t] = np.array(tmp, dtype=np.float32)

    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="test json path (one sample)")
    ap.add_argument("--artifacts", default="artifacts", help="artifact dir")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    art = args.artifacts
    pt_path = os.path.join(art, "attn_lstm.pt")
    scaler_path = os.path.join(art, "attn_scaler.npz")
    labels_path = os.path.join(art, "labels.json")
    cfg_path = os.path.join(art, "attn_config.json")
    labels_ko_path = os.path.join(art, "labels_ko.json")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"missing: {pt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"missing: {scaler_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"missing: {labels_path}")

    # labels
    labels_obj = load_json(labels_path)
    classes = labels_obj.get("classes") or labels_obj.get("labels")
    if not isinstance(classes, list) or not classes:
        raise RuntimeError("labels.json must contain {classes:[...]} or {labels:[...]}")

    # optional ko labels
    ko_map = {}
    if os.path.exists(labels_ko_path):
        ko_obj = load_json(labels_ko_path)
        # 허용 포맷:
        # 1) {"map":{"yes":"네",...}}
        # 2) {"yes":"네", ...}
        if isinstance(ko_obj, dict) and "map" in ko_obj and isinstance(ko_obj["map"], dict):
            ko_map = ko_obj["map"]
        elif isinstance(ko_obj, dict):
            ko_map = ko_obj

    # scaler
    sc = np.load(scaler_path)
    mean = sc["mean"].astype(np.float32)
    scale = sc["scale"].astype(np.float32)
    D = int(mean.shape[0])

    # config (없으면 최대한 추정)
    T = 60
    hidden = 128
    layers = 2
    dropout = 0.35
    counts = {"leftHand": 21, "rightHand": 21, "pose": 25, "face": 70}

    if os.path.exists(cfg_path):
        cfg = load_json(cfg_path)
        T = int(cfg.get("T", T))
        hidden = int(cfg.get("hidden", hidden))
        layers = int(cfg.get("layers", layers))
        dropout = float(cfg.get("dropout", dropout))
        if isinstance(cfg.get("counts"), dict):
            counts.update(cfg["counts"])

    # load sample
    obj = load_json(args.json)

    # build X (T,D) then standardize
    X = build_X_from_collect_json(obj, T=T, D=D, counts=counts)
    Xs = (X - mean[None, :]) / np.maximum(scale[None, :], 1e-8)

    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttnBiLSTMClassifier(
        input_dim=D,
        hidden_dim=hidden,
        num_layers=layers,
        num_classes=len(classes),
        dropout=dropout,
    ).to(device)

    state = torch.load(pt_path, map_location=device)
    # 학습 스크립트가 state_dict만 저장했으니 그대로 로드
    model.load_state_dict(state)
    model.eval()

    xb = torch.from_numpy(Xs[None, :, :]).float().to(device)  # (1,T,D)
    with torch.no_grad():
        logits = model(xb)[0]  # (C,)
        prob = torch.softmax(logits, dim=0).cpu().numpy()

    topk = min(args.topk, len(classes))
    idxs = np.argsort(-prob)[:topk]

    # print
    print("=== RESULT ===")
    print(f"json: {args.json}")
    # 만약 json에 라벨이 들어있으면 참고로 출력
    gt = obj.get("label") or obj.get("label_key") or obj.get("class") or obj.get("y")
    if gt is not None:
        print(f"gt(label in json): {gt}")

    for rank, i in enumerate(idxs, start=1):
        key = str(classes[i])
        ko = ko_map.get(key)
        p = float(prob[i])
        if ko:
            print(f"{rank:02d}. {key} / {ko}  prob={p:.4f}")
        else:
            print(f"{rank:02d}. {key}  prob={p:.4f}")


if __name__ == "__main__":
    main()
