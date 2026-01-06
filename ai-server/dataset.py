import json
from pathlib import Path
import numpy as np
from normalize import normalize_frame

# ---- helpers ----
def _reshape_flat_kps(flat, n_points, name="kps"):
    """
    flat: [x,y,c, x,y,c, ...]
    returns (n_points,3) float32, pad/trim safe
    """
    if flat is None:
        return None
    arr = np.asarray(flat, dtype=np.float32).reshape(-1)
    need = n_points * 3
    if arr.size < need:
        pad = np.zeros((need - arr.size,), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.size > need:
        arr = arr[:need]
    return arr.reshape(n_points, 3)

def _points_from_frame_dict(fr: dict):
    """
    React/일반 frames 구조: fr = {pose:[{x,y,confidence}..], leftHand:..., rightHand:..., face:...}
    """
    def to_np(points, n):
        if points is None:
            return None

        # case A) [{x,y,confidence}, ...]
        if isinstance(points, list) and len(points) > 0 and isinstance(points[0], dict):
            out = np.zeros((n, 3), dtype=np.float32)
            for i in range(min(n, len(points))):
                p = points[i]
                out[i, 0] = float(p.get("x", 0))
                out[i, 1] = float(p.get("y", 0))
                out[i, 2] = float(p.get("confidence", p.get("score", 0)))
            return out

        # case B) [[x,y,c], ...]
        if isinstance(points, list) and len(points) > 0 and isinstance(points[0], (list, tuple)):
            out = np.zeros((n, 3), dtype=np.float32)
            for i in range(min(n, len(points))):
                p = points[i]
                if len(p) > 0: out[i, 0] = float(p[0])
                if len(p) > 1: out[i, 1] = float(p[1])
                if len(p) > 2: out[i, 2] = float(p[2])
            return out

        # case C) flat [x,y,c, x,y,c...]
        if isinstance(points, list):
            return _reshape_flat_kps(points, n)

        return None

    pose = to_np(fr.get("pose") or fr.get("Pose"), 25)
    lh   = to_np(fr.get("leftHand") or fr.get("left_hand") or fr.get("left") or fr.get("LeftHand"), 21)
    rh   = to_np(fr.get("rightHand") or fr.get("right_hand") or fr.get("right") or fr.get("RightHand"), 21)
    face = to_np(fr.get("face") or fr.get("Face"), 70)
    return pose, lh, rh, face

def _extract_seq_from_aihub_openpose_style(obj: dict):
    """
    네가 준 샘플 구조:
    {
      "people": { "pose_keypoints_2d":[...75...], "hand_left_keypoints_2d":[...63...], ... }
    }
    => 이 경우 '한 파일이 1프레임'인 걸로 보고 T=1 반환
    """
    people = obj.get("people")
    if people is None:
        return None

    # people이 list일 수도 있고 dict일 수도 있음 -> 첫 사람만
    if isinstance(people, list):
        if len(people) == 0:
            return None
        p = people[0]
    else:
        p = people

    pose2d = p.get("pose_keypoints_2d")
    lh2d   = p.get("hand_left_keypoints_2d")
    rh2d   = p.get("hand_right_keypoints_2d")
    face2d = p.get("face_keypoints_2d")

    pose = _reshape_flat_kps(pose2d, 25, "pose")
    lh   = _reshape_flat_kps(lh2d, 21, "lh")
    rh   = _reshape_flat_kps(rh2d, 21, "rh")
    face = _reshape_flat_kps(face2d, 70, "face")

    feat = normalize_frame(pose, lh, rh, face)  # (411,)
    return np.stack([feat], axis=0)             # (1,411)

def parse_one_json(path: Path):
    """
    반환: (T,411)
    지원 케이스:
      A) {frames:[...]}  (프레임 리스트)
      B) {people:{...keypoints_2d flat...}} (네가 준 샘플)
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    # Case A) frames list
    frames = obj.get("frames") or obj.get("data") or obj.get("sequence") or obj.get("keypoints")
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], dict):
        seq = []
        for fr in frames:
            pose, lh, rh, face = _points_from_frame_dict(fr)
            feat = normalize_frame(pose, lh, rh, face)
            seq.append(feat)
        return np.stack(seq, axis=0).astype(np.float32)

    # Case B) openpose-style
    seq = _extract_seq_from_aihub_openpose_style(obj)
    if seq is not None:
        return seq.astype(np.float32)

    raise ValueError("Unsupported JSON structure (no frames list, no people keypoints_2d).")
