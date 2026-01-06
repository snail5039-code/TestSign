from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Any
import numpy as np
import torch
import json
from pathlib import Path

from model import AttnBiGRU
from normalize import normalize_frame
from feature import add_delta

MAX_FRAMES = 30

app = FastAPI()

class FrameIn(BaseModel):
    pose: Optional[List[Any]] = None
    leftHand: Optional[List[Any]] = None
    rightHand: Optional[List[Any]] = None
    face: Optional[List[Any]] = None

class PredictIn(BaseModel):
    frames: List[FrameIn]

def to_np(points, expected):
    if points is None:
        return None

    # [{x,y,confidence}]
    if isinstance(points, list) and len(points) > 0 and isinstance(points[0], dict):
        out = np.zeros((expected,3), dtype=np.float32)
        for i in range(min(expected, len(points))):
            p = points[i]
            out[i,0] = float(p.get("x", 0))
            out[i,1] = float(p.get("y", 0))
            out[i,2] = float(p.get("confidence", p.get("score", 0)))
        return out

    # [[x,y,c]]
    if isinstance(points, list) and len(points) > 0 and isinstance(points[0], (list, tuple)):
        out = np.zeros((expected,3), dtype=np.float32)
        for i in range(min(expected, len(points))):
            p = points[i]
            if len(p) > 0: out[i,0] = float(p[0])
            if len(p) > 1: out[i,1] = float(p[1])
            if len(p) > 2: out[i,2] = float(p[2])
        return out

    # flat [x,y,c,...]
    if isinstance(points, list):
        arr = np.asarray(points, dtype=np.float32).reshape(-1)
        need = expected * 3
        if arr.size < need:
            arr = np.concatenate([arr, np.zeros((need-arr.size,), dtype=np.float32)], axis=0)
        else:
            arr = arr[:need]
        return arr.reshape(expected,3)

    return None

# ---- load model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = Path("weights.pt")
if not weights_path.exists():
    raise RuntimeError("weights.pt not found. Run train.py first.")

ckpt = torch.load(weights_path, map_location=device)
classes = int(ckpt["classes"])
model = AttnBiGRU(classes=classes).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

label_map_path = Path("label_map.json")
inv = {}
if label_map_path.exists():
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    inv = {int(v): k for k, v in label_map.items()}

word_table_path = Path("word_table.json")
word_table = {}
if word_table_path.exists():
    word_table = json.loads(word_table_path.read_text(encoding="utf-8"))

@app.get("/health")
def health():
    return {"ok": True, "frames": MAX_FRAMES, "classes": classes, "device": device}

@app.post("/predict")
def predict(inp: PredictIn):
    frames = inp.frames[-MAX_FRAMES:]

    seq = []
    for fr in frames:
        pose = to_np(fr.pose, 25)
        lh   = to_np(fr.leftHand, 21)
        rh   = to_np(fr.rightHand, 21)
        face = to_np(fr.face, 70)
        seq.append(normalize_frame(pose, lh, rh, face))  # (411,)

    # 앞쪽 0으로 padding 해서 30 맞춤
    if len(seq) < MAX_FRAMES:
        pad = [np.zeros((411,), dtype=np.float32)] * (MAX_FRAMES - len(seq))
        seq = pad + seq

    seq_411 = np.stack(seq, axis=0).astype(np.float32)  # (30,411)
    seq_822 = add_delta(seq_411)                        # (30,822)

    xt = torch.from_numpy(seq_822[None, ...]).to(device)  # (1,30,822)

    with torch.no_grad():
        logits = model(xt)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls = int(prob.argmax())
        conf = float(prob[cls])

    label = inv.get(cls, str(cls))
    text = word_table.get(label, label)

    top5 = prob.argsort()[-5:][::-1]
    candidates = [(inv.get(int(i), str(int(i))), float(prob[int(i)])) for i in top5]

    return {"label": label, "text": text, "confidence": conf, "candidates": candidates}
