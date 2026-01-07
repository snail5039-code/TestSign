# app/server.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.attn_classifier import AttnLSTMRunner
from .template_classifier import seq_to_matrix  # (left,right,pose,face) + xyz

# ----------------------------
# Artifact dirs (1H / 2H)
# ----------------------------
ART_1H = Path("artifacts_1h")
ART_2H = Path("artifacts_2h")

def _paths(art_dir: Path):
    return (
        art_dir / "attn_lstm.pt",
        art_dir / "attn_scaler.npz",
        art_dir / "labels.json",
        art_dir / "attn_config.json",
    )

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Sign Translation API (Auto)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner_1h: Optional[AttnLSTMRunner] = None
runner_2h: Optional[AttnLSTMRunner] = None


class Hint(BaseModel):
    # 프론트가 보내줘도 되고, 안 보내도 서버가 frames로 다시 계산함
    maxHandsSeen: Optional[int] = None
    bothRatio: Optional[float] = None


class PredictRequest(BaseModel):
    frames: List[Dict[str, Any]] = Field(..., description="frames: list of {leftHand,rightHand,pose,face}")
    hint: Optional[Hint] = None


@app.on_event("startup")
def _startup():
    global runner_1h, runner_2h

    # 1H
    m, s, l, c = _paths(ART_1H)
    miss = [str(p) for p in [m, s, l, c] if not p.exists()]
    if miss:
        raise RuntimeError(f"[1H] Missing artifacts: {miss}")
    runner_1h = AttnLSTMRunner(model_path=m, scaler_path=s, labels_path=l, config_path=c, device="cpu")

    # 2H
    m2, s2, l2, c2 = _paths(ART_2H)
    miss2 = [str(p) for p in [m2, s2, l2, c2] if not p.exists()]
    if miss2:
        raise RuntimeError(f"[2H] Missing artifacts: {miss2}")
    runner_2h = AttnLSTMRunner(model_path=m2, scaler_path=s2, labels_path=l2, config_path=c2, device="cpu")


@app.get("/health")
def health():
    assert runner_1h is not None and runner_2h is not None
    return {
        "ok": True,
        "1h": {"T": runner_1h.T, "D": runner_1h.D, "art": str(ART_1H)},
        "2h": {"T": runner_2h.T, "D": runner_2h.D, "art": str(ART_2H)},
    }


# ----------------------------
# Auto routing helpers
# ----------------------------
def _is_nonzero_vec(v: Any, eps: float = 1e-8) -> bool:
    if not isinstance(v, list) or len(v) == 0:
        return False
    # 빠르게 0인지 검사(대부분은 0이 길게 이어짐)
    # abs(v) > eps 가 하나라도 있으면 True
    for x in v:
        if isinstance(x, (int, float)) and abs(float(x)) > eps:
            return True
    return False


def compute_both_ratio(frames: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    bothRatio = (left nonzero AND right nonzero) / T
    anyRatio  = (left nonzero OR  right nonzero) / T
    """
    if not frames:
        return {"bothRatio": 0.0, "anyRatio": 0.0}

    both = 0
    anyh = 0
    T = len(frames)
    for fr in frames:
        lh = _is_nonzero_vec((fr or {}).get("leftHand"))
        rh = _is_nonzero_vec((fr or {}).get("rightHand"))
        if lh or rh:
            anyh += 1
        if lh and rh:
            both += 1
    return {
        "bothRatio": both / T,
        "anyRatio": anyh / T,
    }


def choose_mode(frames: List[Dict[str, Any]], hint: Optional[Hint]) -> str:
    """
    선택 규칙(실전형):
      - 손이 거의 안 잡히면 (anyRatio 낮음) -> 1h로 보내도 의미 없으니 400 처리
      - 둘 다 비율이 어느 정도면 -> 2h
      - 아니면 -> 1h
    """
    stats = compute_both_ratio(frames)
    both_ratio = stats["bothRatio"]
    any_ratio = stats["anyRatio"]

    # 힌트가 있으면 보정(있어도 서버 계산이 우선)
    if hint is not None:
        if hint.bothRatio is not None and np.isfinite(hint.bothRatio):
            both_ratio = max(both_ratio, float(hint.bothRatio))
        # maxHandsSeen이 2면 2h 쪽으로 가중
        if hint.maxHandsSeen == 2:
            both_ratio = max(both_ratio, 0.25)

    if any_ratio < 0.10:
        # 손이 10% 미만 프레임에서만 보이면 캡처가 실패한 케이스
        raise HTTPException(status_code=400, detail=f"Hand not detected enough (anyRatio={any_ratio:.3f})")

    # 2손이 20% 이상 프레임에서 동시에 잡히면 2H로 간주
    return "2h" if both_ratio >= 0.20 else "1h"


def run_predict(runner: AttnLSTMRunner, frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    # seq_to_matrix가 (T,411) 만들어줌. T는 runner.T로 맞춰준다.
    X = seq_to_matrix(frames, T=runner.T)
    out = runner.predict(X)
    return out


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/predict/1h")
def predict_1h(req: PredictRequest):
    assert runner_1h is not None
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")

    out = run_predict(runner_1h, req.frames)
    return {
        "mode": "1h",
        "label": out["label"],
        "confidence": float(out["confidence"]),
        "text": out["label"],
        "top5": out["top5"],
        "pred_idx": out.get("pred_idx"),
    }


@app.post("/predict/2h")
def predict_2h(req: PredictRequest):
    assert runner_2h is not None
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")

    out = run_predict(runner_2h, req.frames)
    return {
        "mode": "2h",
        "label": out["label"],
        "confidence": float(out["confidence"]),
        "text": out["label"],
        "top5": out["top5"],
        "pred_idx": out.get("pred_idx"),
    }


@app.post("/predict/auto")
def predict_auto(req: PredictRequest):
    assert runner_1h is not None and runner_2h is not None
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames is empty")

    mode = choose_mode(req.frames, req.hint)
    stats = compute_both_ratio(req.frames)

    runner = runner_2h if mode == "2h" else runner_1h
    out = run_predict(runner, req.frames)

    return {
        "mode": mode,
        "label": out["label"],
        "confidence": float(out["confidence"]),
        "text": out["label"],
        "top5": out["top5"],
        "pred_idx": out.get("pred_idx"),
        "debug": {
            "bothRatio": stats["bothRatio"],
            "anyRatio": stats["anyRatio"],
            "T_in": len(req.frames),
            "T_model": runner.T,
            "D_model": runner.D,
        },
    }
