import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

DATASET_DIR = Path("dataset")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 네 JSON에 실제로 존재하는 키들
MODALITIES = ["leftHand", "rightHand", "pose", "face"]

def detect_frames(obj):
    # {"frames":[...]} 형태
    if isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
        return obj["frames"]
    # 이미 frames 리스트인 경우
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (dict, list)):
        return obj
    # 단일 frame dict
    if isinstance(obj, dict):
        return [obj]
    return []

def _get_xyz_from_dict(d):
    # 다양한 키 케이스 대응 (x/X, y/Y, z/Z)
    x = d.get("x", d.get("X", 0.0))
    y = d.get("y", d.get("Y", 0.0))
    z = d.get("z", d.get("Z", 0.0))
    return float(x), float(y), float(z)

def parse_landmarks(v):
    """
    반환: np.ndarray shape (N,3)
    지원 형태:
      1) [ {"x":..,"y":..,"z":..}, ... ]
      2) [ [x,y,z], [x,y,z], ... ]
      3) [x1,y1,z1,x2,y2,z2,...]  (flat)
      4) 길이가 4의 배수(flat)인 경우: [x,y,z,score, x,y,z,score,...] -> score 버림
      5) dict에 landmarks 키가 있는 경우
    """
    if v is None:
        return np.zeros((0, 3), dtype=np.float32)

    # dict 래핑 케이스
    if isinstance(v, dict):
        if "landmarks" in v:
            return parse_landmarks(v["landmarks"])
        # 단일 랜드마크 dict일 수도 있음
        if ("x" in v or "X" in v) and ("y" in v or "Y" in v):
            x, y, z = _get_xyz_from_dict(v)
            return np.array([[x, y, z]], dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    # list/tuple 케이스
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        first = v[0]

        # 1) list of dict
        if isinstance(first, dict):
            rows = []
            for d in v:
                if not isinstance(d, dict):
                    continue
                if ("x" in d or "X" in d) and ("y" in d or "Y" in d):
                    rows.append(_get_xyz_from_dict(d))
            if len(rows) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array(rows, dtype=np.float32)

        # 2) list of list/tuple
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            rows = []
            for p in v:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = float(p[0])
                y = float(p[1])
                z = float(p[2]) if len(p) > 2 else 0.0
                rows.append((x, y, z))
            if len(rows) == 0:
                return np.zeros((0, 3), dtype=np.float32)
            return np.array(rows, dtype=np.float32)

        # 3) flat list of numbers
        if isinstance(first, (int, float)):
            arr = np.array(v, dtype=np.float32)
            if arr.size % 3 == 0:
                return arr.reshape(-1, 3)
            if arr.size % 4 == 0:
                return arr.reshape(-1, 4)[:, :3]
            # 애매하면 최대한 3개씩 잘라서 사용
            n = arr.size // 3
            if n <= 0:
                return np.zeros((0, 3), dtype=np.float32)
            return arr[: n * 3].reshape(-1, 3)

    return np.zeros((0, 3), dtype=np.float32)

def pad_truncate_landmarks(xyz, target_n):
    if target_n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    n = xyz.shape[0]
    if n == target_n:
        return xyz.astype(np.float32, copy=False)
    out = np.zeros((target_n, 3), dtype=np.float32)
    m = min(n, target_n)
    if m > 0:
        out[:m] = xyz[:m]
    return out

def pad_truncate_frames(frames, target_t):
    if target_t <= 0:
        return frames
    if len(frames) == target_t:
        return frames
    if len(frames) > target_t:
        return frames[:target_t]
    # 부족하면 마지막 프레임을 반복(없으면 빈 dict)
    last = frames[-1] if len(frames) > 0 else {}
    return frames + [last] * (target_t - len(frames))

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Not found: {DATASET_DIR.resolve()}")

    class_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    if not class_names:
        raise RuntimeError("No class folders found under dataset/")

    # 모든 파일 수집
    samples = []
    for cdir in class_dirs:
        for fp in sorted(cdir.glob("*.json")):
            samples.append((cdir.name, fp))
    if not samples:
        raise RuntimeError("No .json files found.")

    print(f"[OK] classes={len(class_names)} samples={len(samples)}")

    # 1) 시퀀스 길이(T) 모드(최빈값) 결정
    t_counter = Counter()
    # 2) modality별 landmark 개수의 모드(최빈값) 결정
    count_counter = {m: Counter() for m in MODALITIES}

    # 빠르게 전체 스캔(350개면 부담 없음)
    for _, fp in samples:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        frames = detect_frames(obj)
        t_counter[len(frames)] += 1

        # 프레임 일부만 추출해서 count 추정(속도)
        for fr in frames[:5]:
            if not isinstance(fr, dict):
                continue
            for m in MODALITIES:
                xyz = parse_landmarks(fr.get(m))
                count_counter[m][xyz.shape[0]] += 1

    T = t_counter.most_common(1)[0][0]
    # 0을 제외한 최빈값을 우선, 없으면 0
    expected_counts = {}
    for m in MODALITIES:
        nonzero = [(k, v) for k, v in count_counter[m].items() if k > 0]
        if nonzero:
            expected_counts[m] = sorted(nonzero, key=lambda kv: kv[1], reverse=True)[0][0]
        else:
            expected_counts[m] = 0

    print(f"[INFO] T(mode)={T}  T_dist_top={t_counter.most_common(5)}")
    print("[INFO] expected landmark counts (mode, nonzero-preferred):")
    for m in MODALITIES:
        print(f"  - {m}: {expected_counts[m]} (raw top={count_counter[m].most_common(3)})")

    # 최종 feature dim
    D = sum(expected_counts[m] * 3 for m in MODALITIES)
    if D <= 0:
        raise RuntimeError("Feature dimension D is 0. Parsing likely failed; inspect one JSON structure.")

    label_to_idx = {name: i for i, name in enumerate(class_names)}

    X = np.zeros((len(samples), T, D), dtype=np.float32)
    y = np.zeros((len(samples),), dtype=np.int64)

    for i, (label, fp) in enumerate(samples):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        frames = detect_frames(obj)
        frames = pad_truncate_frames(frames, T)

        seq = np.zeros((T, D), dtype=np.float32)

        for t, fr in enumerate(frames):
            if not isinstance(fr, dict):
                fr = {}
            feats = []
            for m in MODALITIES:
                target_n = expected_counts[m]
                xyz = parse_landmarks(fr.get(m))
                xyz = pad_truncate_landmarks(xyz, target_n)  # (N,3)
                feats.append(xyz.reshape(-1))  # (N*3,)
            vec = np.concatenate(feats, axis=0) if feats else np.zeros((D,), dtype=np.float32)
            # 안전장치
            if vec.shape[0] != D:
                vv = np.zeros((D,), dtype=np.float32)
                mm = min(D, vec.shape[0])
                vv[:mm] = vec[:mm]
                vec = vv
            seq[t] = vec

        X[i] = seq
        y[i] = label_to_idx[label]

        if (i + 1) % 50 == 0:
            print(f"[PROGRESS] {i+1}/{len(samples)}")

    out_npz = OUT_DIR / "dataset.npz"
    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        class_names=np.array(class_names),
        T=np.int32(T),
        D=np.int32(D),
        leftHand_count=np.int32(expected_counts["leftHand"]),
        rightHand_count=np.int32(expected_counts["rightHand"]),
        pose_count=np.int32(expected_counts["pose"]),
        face_count=np.int32(expected_counts["face"]),
    )
    print(f"[DONE] Saved: {out_npz.resolve()}")
    print(f"[SHAPE] X={X.shape} y={y.shape}  (T={T}, D={D})")

if __name__ == "__main__":
    main()
