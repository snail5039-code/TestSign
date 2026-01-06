# app/server.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.attn_classifier import AttnLSTMRunner
from .template_classifier import TemplateClassifier, seq_to_matrix

ARTIFACTS_DIR = Path("artifacts")

TEMPLATES_PATH = ARTIFACTS_DIR / "templates.npz"

ATTN_MODEL_PATH = ARTIFACTS_DIR / "attn_lstm.pt"
ATTN_SCALER_PATH = ARTIFACTS_DIR / "attn_scaler.npz"
ATTN_LABELS_PATH = ARTIFACTS_DIR / "labels.json"
ATTN_CONFIG_PATH = ARTIFACTS_DIR / "attn_config.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용. 배포 시 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

template_clf: Optional[TemplateClassifier] = None
attn_runner: Optional[AttnLSTMRunner] = None


class PredictRequest(BaseModel):
    frames: List[Dict[str, Any]] = Field(
        ...,
        description="List of frame dicts with keys: leftHand/rightHand/pose/face",
    )


@app.on_event("startup")
def _startup():
    global template_clf, attn_runner

    # (1) template classifier는 있으면 로드(옵션)
    if TEMPLATES_PATH.exists():
        template_clf = TemplateClassifier.load(TEMPLATES_PATH)
    else:
        template_clf = None

    # (2) AttnLSTM (필수)
    missing = []
    for p in [ATTN_MODEL_PATH, ATTN_SCALER_PATH, ATTN_LABELS_PATH, ATTN_CONFIG_PATH]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise RuntimeError(f"Missing artifacts: {missing}")

    # 여기서 Runner가 체크포인트 구조를 자동으로 맞춰서 로드함(BiLSTM 포함)
    attn_runner = AttnLSTMRunner(
        model_path=ATTN_MODEL_PATH,
        scaler_path=ATTN_SCALER_PATH,
        labels_path=ATTN_LABELS_PATH,
        config_path=ATTN_CONFIG_PATH,
        device="cpu",
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "templates": str(TEMPLATES_PATH) if TEMPLATES_PATH.exists() else None,
        "attn_model": str(ATTN_MODEL_PATH),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    global attn_runner

    if not isinstance(req.frames, list) or len(req.frames) == 0:
        raise HTTPException(status_code=400, detail="frames is empty")

    # frames -> (30, 411)
    X = seq_to_matrix(req.frames, T=30)

    out = attn_runner.predict(X)

    return {
        "label": out["label"],
        "confidence": out["confidence"],
        "text": out["label"],
        "top5": out["top5"],
        "pred_idx": out.get("pred_idx"),
    }
