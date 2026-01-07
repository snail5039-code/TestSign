from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# 고정 랜드마크 개수 (네 데이터에서 mode로 확인된 값)
N_LEFT = 21
N_RIGHT = 21
N_POSE = 25
N_FACE = 70

DIMS_PER_LM = 3  # x,y,z
D_TOTAL = (N_LEFT + N_RIGHT + N_POSE + N_FACE) * DIMS_PER_LM  # 411


def _landmarks_to_flat(landmarks: Any, expected_n: int) -> np.ndarray:
    """
    landmarks는 아래 중 하나일 수 있음:
      - None / missing
      - List[{"x":..., "y":..., "z":...}, ...]
      - List[[x,y,z], ...]
    반환: (expected_n*3,) float32 (부족하면 0 padding, 넘치면 truncate)
    """
    out = np.zeros((expected_n, 3), dtype=np.float32)

    if landmarks is None:
        return out.reshape(-1)

    if not isinstance(landmarks, list):
        return out.reshape(-1)

    # 요소가 dict인지 list/tuple인지에 따라 파싱
    for i in range(min(expected_n, len(landmarks))):
        p = landmarks[i]
        if isinstance(p, dict):
            out[i, 0] = float(p.get("x", 0.0))
            out[i, 1] = float(p.get("y", 0.0))
            out[i, 2] = float(p.get("z", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            out[i, 0] = float(p[0])
            out[i, 1] = float(p[1])
            out[i, 2] = float(p[2])
        else:
            # 알 수 없는 포맷이면 0 유지
            pass

    return out.reshape(-1)


def frame_to_vec(frame: Dict[str, Any]) -> np.ndarray:
    """
    frame: {"leftHand": [...], "rightHand": [...], "pose": [...], "face": [...]}
    반환: (411,) float32
    템플릿/데이터셋 생성 때와 동일한 순서로 이어붙여야 함:
      leftHand -> rightHand -> pose -> face
    """
    lh = _landmarks_to_flat(frame.get("leftHand"), N_LEFT)
    rh = _landmarks_to_flat(frame.get("rightHand"), N_RIGHT)
    pose = _landmarks_to_flat(frame.get("pose"), N_POSE)
    face = _landmarks_to_flat(frame.get("face"), N_FACE)

    v = np.concatenate([lh, rh, pose, face], axis=0).astype(np.float32)
    if v.shape[0] != D_TOTAL:
        # 혹시 모를 방어
        vv = np.zeros((D_TOTAL,), dtype=np.float32)
        n = min(D_TOTAL, v.shape[0])
        vv[:n] = v[:n]
        return vv
    return v


def seq_to_matrix(frames: List[Dict[str, Any]], T: int = 30) -> np.ndarray:
    """
    frames 길이가 30이 아니면:
      - 길면 앞에서부터 T개 truncate
      - 짧으면 뒤를 0프레임으로 padding
    반환: (T, 411)
    """
    X = np.zeros((T, D_TOTAL), dtype=np.float32)
    n = min(T, len(frames))
    for i in range(n):
        X[i] = frame_to_vec(frames[i])
    return X


@dataclass
class TemplateClassifier:
    templates: np.ndarray  # (C,T,D)
    class_names: np.ndarray  # (C,)

    @classmethod
    def load(cls, templates_npz_path: str | Path) -> "TemplateClassifier":
        data = np.load(str(templates_npz_path), allow_pickle=True)
        templates = data["templates"].astype(np.float32)
        class_names = data["class_names"]
        return cls(templates=templates, class_names=class_names)

    def predict(self, X: np.ndarray) -> Tuple[str, int, float, List[Tuple[str, float]]]:
        """
        X: (T,D)
        return: (pred_name, pred_idx, pred_dist, top5[(name,dist)])
        """
        # MSE distance: mean over (T,D)
        dists = np.mean((self.templates - X[None, :, :]) ** 2, axis=(1, 2))  # (C,)
        pred_idx = int(np.argmin(dists))
        pred_name = str(self.class_names[pred_idx])
        pred_dist = float(dists[pred_idx])

        order = np.argsort(dists)[:5]
        top5 = [(str(self.class_names[int(i)]), float(dists[int(i)])) for i in order]
        return pred_name, pred_idx, pred_dist, top5
