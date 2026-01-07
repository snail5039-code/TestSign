from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .attn_classifier import AttnLSTMRunner

# ----------------------------
# Paths (env override 가능)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # ai-server/
ART_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))

MODEL_PATH = Path(os.getenv("ATTN_MODEL", str(ART_DIR / "attn_lstm.pt")))
SCALER_PATH = Path(os.getenv("ATTN_SCALER", str(ART_DIR / "attn_scaler.npz")))
LABELS_PATH = Path(os.getenv("ATTN_LABELS", str(ART_DIR / "labels.json")))
CONFIG_PATH = Path(os.getenv("ATTN_CONFIG", str(ART_DIR / "attn_config.json")))

# expected fixed dims (프론트 기준)
POSE_LEN = 25 * 3   # 75
FACE_LEN = 70 * 3   # 210
HAND_LEN = 21 * 3   # 63

app = FastAPI(title="Sign Translation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    frames: List[Dict[str, Any]]

RUNNER: Optional[AttnLSTMRunner] = None

@app.on_event("startup")
def _startup():
    global RUNNER
    RUNNER = AttnLSTMRunner(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        labels_path=LABELS_PATH,
        config_path=CONFIG_PATH,
        device=os.getenv("DEVICE", "cpu"),
    )

@app.get("/health")
def health():
    return {"ok": True}

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)

def _coerce_flat(val: Any, expected_len: int, point_count: int) -> List[float]:
    """
    val이 아래 중 무엇이든 expected_len 길이의 float list로 변환:
      1) flat 숫자 배열: [x,y,c,x,y,c,...]
      2) landmarks list: [{x,y,z or visibility/presence}, ...]
      3) landmarks list: [[x,y,z?], ...]
      4) dict wrapper: {"landmarks":[...]} / {"points":[...]}
    실패하면 0으로 채움.
    """
    if val is None:
        return [0.0] * expected_len

    if isinstance(val, dict):
        if "landmarks" in val:
            val = val["landmarks"]
        elif "points" in val:
            val = val["points"]

    # already flat numeric (exact)
    if isinstance(val, list) and len(val) == expected_len and all(_is_number(x) for x in val):
        return [float(x) for x in val]

    # flat numeric but length mismatch (tolerant)
    if isinstance(val, list) and len(val) > 0 and all(_is_number(x) for x in val):
        arr = [float(x) for x in val]
        if len(arr) >= expected_len:
            return arr[:expected_len]
        return arr + [0.0] * (expected_len - len(arr))

    # landmarks list -> flatten
    if isinstance(val, list) and len(val) == point_count:
        out: List[float] = []
        for p in val:
            if isinstance(p, dict):
                x = float(p.get("x", 0.0))
                y = float(p.get("y", 0.0))
                z = p.get("z", p.get("visibility", p.get("presence", 0.0)))
                z = float(z) if _is_number(z) else 0.0
                out.extend([x, y, z])
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                x = float(p[0])
                y = float(p[1])
                z = float(p[2]) if (len(p) >= 3 and _is_number(p[2])) else 0.0
                out.extend([x, y, z])
            else:
                out.extend([0.0, 0.0, 0.0])

        if len(out) == expected_len:
            return out

    return [0.0] * expected_len

def _get_any(frame: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in frame:
            return frame[k]
    return None

def frames_to_matrix(frames: List[Dict[str, Any]], T: int, D: int) -> np.ndarray:
    """
    IMPORTANT:
      학습(build_dataset.py) 순서가 leftHand -> rightHand -> pose -> face 였으므로,
      서버도 반드시 동일 순서로 concat해야 scaler/모델과 정합됨.
    """
    X = np.zeros((T, D), dtype=np.float32)

    n = min(len(frames), T)
    for i in range(n):
        fr = frames[i] or {}

        left_val = _get_any(fr, ["leftHand", "left_hand", "left", "handLeft"])
        right_val = _get_any(fr, ["rightHand", "right_hand", "right", "handRight"])
        pose_val = _get_any(fr, ["pose", "poseLandmarks", "pose_landmarks"])
        face_val = _get_any(fr, ["face", "faceLandmarks", "face_landmarks"])

        left = _coerce_flat(left_val, HAND_LEN, 21)
        right = _coerce_flat(right_val, HAND_LEN, 21)
        pose = _coerce_flat(pose_val, POSE_LEN, 25)
        face = _coerce_flat(face_val, FACE_LEN, 70)

        # ★ 학습과 동일 concat 순서
        vec = np.array(left + right + pose + face, dtype=np.float32)

        if vec.size != D:
            if vec.size > D:
                vec = vec[:D]
            else:
                vec = np.concatenate([vec, np.zeros((D - vec.size,), dtype=np.float32)], axis=0)

        X[i] = vec

    return X

@app.post("/predict")
def predict(req: PredictRequest):
    assert RUNNER is not None, "Runner not initialized"

    X = frames_to_matrix(req.frames, T=RUNNER.T, D=RUNNER.D)

    if os.getenv("DEBUG_INPUT", "0") == "1":
        # 손 영역 평균이 0에 가까우면 사실상 손이 안 들어온 것
        left_right = X[:, : (63 + 63)]
        print(
            f"[DEBUG_INPUT] X: min={float(X.min()):.6f} max={float(X.max()):.6f} mean={float(X.mean()):.6f} "
            f"| hands_abs_mean={float(np.abs(left_right).mean()):.6f}"
        )

    out = RUNNER.predict(X)

    label = out.get("label") or out.get("pred") or "unknown"
    conf = float(out.get("confidence", 0.0))
    top5 = out.get("top5", [])

    return {
        "label": label,
        "confidence": conf,
        "text": label,
        "top5": top5,
        "pred_idx": out.get("pred_idx", None),
    }
