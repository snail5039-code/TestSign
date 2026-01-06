# app/attn_classifier.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# basic utils
# -----------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_key(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def _strip_known_prefixes(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["module.", "net.", "model.", "classifier."]
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    raise ValueError("Unsupported checkpoint format (expected state_dict or {'state_dict':...})")


def _has_key(sd: Dict[str, torch.Tensor], name: str) -> bool:
    # 대소문자 구분 없이 확인하도록 수정
    name = name.lower()
    if name in sd:
        return True
    suf = "." + name
    return any(k.endswith(suf) for k in sd.keys())


def _get_key(sd: Dict[str, torch.Tensor], name: str) -> str:
    name = name.lower()
    if name in sd:
        return name
    suf = "." + name
    for k in sd.keys():
        if k.endswith(suf):
            return k
    raise KeyError(f"Key not found: {name}")


# -----------------------------
# labels / scaler
# -----------------------------
def _load_labels(path: Path) -> List[str]:
    obj = _load_json(path)

    if isinstance(obj, list):
        return [str(x) for x in obj]

    if isinstance(obj, dict):
        for key in ["classes", "labels", "class_names", "classnames", "label_names"]:
            v = obj.get(key)
            if isinstance(v, list):
                return [str(x) for x in v]

        if all(str(k).isdigit() for k in obj.keys()) and all(isinstance(v, (str, int)) for v in obj.values()):
            n = len(obj)
            return [str(obj[str(i)]) for i in range(n)]

        if all(isinstance(v, (int, float)) for v in obj.values()):
            inv = {int(v): str(k) for k, v in obj.items()}
            return [inv[i] for i in range(len(inv))]

    raise ValueError(f"Unsupported labels.json format: {path}")


def _load_scaler_npz(path: Path, D: int) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)

    candidates = [
        ("mean", "std"), ("mu", "sigma"), ("x_mean", "x_std"),
        ("mean_", "std_"), ("center", "scale"), ("mean", "scale"),
        ("mean_", "scale_"), ("mean_", "scale"), ("mean", "scale_"),
    ]

    mean = std = None
    for a, b in candidates:
        if a in keys and b in keys:
            mean = z[a]
            std = z[b]
            break

    if mean is None or std is None:
        if len(z.files) >= 2:
            mean = z[z.files[0]]
            std = z[z.files[1]]

    if mean is None or std is None:
        raise KeyError(f"Scaler npz keys not recognized. Found keys={sorted(list(keys))}.")

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)

    if mean.size != D:
        mean = mean[:D] if mean.size > D else np.pad(mean, (0, D - mean.size), constant_values=0)
    if std.size != D:
        std = std[:D] if std.size > D else np.pad(std, (0, D - std.size), constant_values=1)

    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32)


# -----------------------------
# infer LSTM prefix/arch robustly
# -----------------------------
def _infer_lstm_prefix(sd: Dict[str, torch.Tensor]) -> str:
    target = "lstm.weight_ih_l0"
    if target in sd:
        return ""
    suf = "." + target
    for k in sd.keys():
        if k.endswith(suf):
            return k[: -len(target)]
    raise ValueError("Checkpoint missing '*lstm.weight_ih_l0'")


def _k(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def _infer_lstm_arch(sd: Dict[str, torch.Tensor]) -> Tuple[int, int, int, bool, str]:
    prefix = _infer_lstm_prefix(sd)
    w0 = sd.get(_k(prefix, "lstm.weight_ih_l0"))
    if w0 is None:
        raise ValueError("Checkpoint missing lstm.weight_ih_l0")

    hidden = int(w0.shape[0] // 4)
    D = int(w0.shape[1])

    layers = 0
    while _k(prefix, f"lstm.weight_ih_l{layers}") in sd:
        layers += 1
    if layers <= 0:
        layers = 1

    bidir = _k(prefix, "lstm.weight_ih_l0_reverse") in sd or any("reverse" in k for k in sd.keys())

    return D, hidden, layers, bidir, prefix


# -----------------------------
# networks
# -----------------------------
class AttnLSTMNet_Simple(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = nn.Linear(out_dim, 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        w = self.attn(h).squeeze(-1)
        a = torch.softmax(w, dim=1).unsqueeze(-1)
        ctx = torch.sum(a * h, dim=1)
        return self.fc(ctx)


class AttnLSTMNet_Head(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
        out_dim: int,
        attn_dim: int,
        head_hidden: int,
        num_classes: int,
        head_dropout: float,
        has_attn_bias: bool = True,  # bias 존재 여부 동적 설정
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # bias=has_attn_bias 설정을 통해 에러 방지
        self.attn_w = nn.Linear(out_dim, attn_dim, bias=has_attn_bias)
        self.attn_v = nn.Linear(attn_dim, 1, bias=has_attn_bias)

        self.head = nn.Sequential(
            nn.Linear(out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=head_dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        u = torch.tanh(self.attn_w(h))
        e = self.attn_v(u).squeeze(-1)
        a = torch.softmax(e, dim=1).unsqueeze(-1)
        ctx = torch.sum(a * h, dim=1)
        return self.head(ctx)


def _find_head_linear_keys(sd: Dict[str, torch.Tensor]) -> Tuple[str, str]:
    pat = re.compile(r"(?:^|\.)(head)\.(\d+)\.weight$")
    idx_keys: List[Tuple[int, str]] = []
    for k in sd.keys():
        m = pat.search(k)
        if m:
            idx = int(m.group(2))
            idx_keys.append((idx, k))
    if not idx_keys:
        raise RuntimeError("No head.<i>.weight keys found in checkpoint")
    idx_keys.sort(key=lambda x: x[0])
    return idx_keys[0][1], idx_keys[-1][1]


@dataclass
class AttnLSTMRunner:
    model_path: Path
    scaler_path: Path
    labels_path: Path
    config_path: Path
    device: str = "cpu"

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.scaler_path = Path(self.scaler_path)
        self.labels_path = Path(self.labels_path)
        self.config_path = Path(self.config_path)

        cfg = _load_json(self.config_path) if self.config_path.exists() else {}
        self.T = int(_first_key(cfg, ["T", "t", "seq_len", "time_steps"], 30))
        self.labels = _load_labels(self.labels_path)

        ckpt = torch.load(self.model_path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        sd = _strip_known_prefixes(sd)
        
        # [핵심] 모든 키를 소문자로 변경하여 대소문자 불일치 해결
        sd = {k.lower(): v for k, v in sd.items()}

        D, hidden, layers, bidir, prefix = _infer_lstm_arch(sd)
        self.D, self.hidden, self.layers, self.bidirectional = int(D), int(hidden), int(layers), bool(bidir)
        out_dim = self.hidden * (2 if self.bidirectional else 1)
        self.mean, self.std = _load_scaler_npz(self.scaler_path, self.D)

        dropout = float(_first_key(cfg, ["dropout"], 0.0))
        head_dropout = float(_first_key(cfg, ["head_dropout", "mlp_dropout"], 0.0))

        # prefix 제거
        if prefix != "":
            sd = { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }

        # 구조 판별
        has_head = _has_key(sd, "attn_w.weight") and any("head." in k for k in sd.keys())
        has_simple = _has_key(sd, "attn.weight") and _has_key(sd, "fc.weight")

        if has_head:
            attn_w_key = _get_key(sd, "attn_w.weight")
            attn_dim = int(sd[attn_w_key].shape[0])
            head_first_key, head_last_key = _find_head_linear_keys(sd)
            head_hidden = int(sd[head_first_key].shape[0])
            num_classes = int(sd[head_last_key].shape[0])

            # [핵심] bias 키가 실제로 있는지 확인하여 모델 생성 시 반영
            has_attn_bias = _has_key(sd, "attn_v.bias") or _has_key(sd, "attn_w.bias")

            self.net = AttnLSTMNet_Head(
                self.D, self.hidden, self.layers, self.bidirectional, dropout,
                out_dim, attn_dim, head_hidden, num_classes, head_dropout,
                has_attn_bias=has_attn_bias
            )
            self.net.load_state_dict(sd, strict=False) # bias 누락 대비 strict=False

        elif has_simple:
            fc_key = _get_key(sd, "fc.weight")
            num_classes = int(sd[fc_key].shape[0])
            self.net = AttnLSTMNet_Simple(self.D, self.hidden, self.layers, num_classes, self.bidirectional, dropout)
            self.net.load_state_dict(sd, strict=True)
        else:
            raise RuntimeError(f"Unknown model head format. Keys: {list(sd.keys())[:10]}")

        self.net.eval().to(self.device)

    @torch.no_grad()
    def predict(self, X: Union[np.ndarray, List[List[float]]]) -> Dict[str, Any]:
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 3: x = x[0]
        
        # T/D 차원 맞춤
        if x.shape[0] < self.T: x = np.vstack([x, np.zeros((self.T - x.shape[0], x.shape[1]), dtype=np.float32)])
        else: x = x[:self.T]
        
        if x.shape[1] != self.D:
            if x.shape[1] > self.D: x = x[:, :self.D]
            else: x = np.hstack([x, np.zeros((x.shape[0], self.D - x.shape[1]), dtype=np.float32)])

        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits = self.net(xt)
        prob = F.softmax(logits, dim=-1).squeeze(0)

        conf, pred_idx = torch.max(prob, dim=-1)
        idx = int(pred_idx.item())
        
        k = min(5, len(self.labels))
        vals, idxs = torch.topk(prob, k=k)
        top5 = [{self.labels[int(i)]: float(v)} for v, i in zip(vals, idxs)]

        return {
            "label": self.labels[idx],
            "confidence": float(conf),
            "top5": top5,
            "pred_idx": idx,
        }