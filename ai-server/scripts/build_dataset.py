import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np

MODALITIES = ["leftHand", "rightHand", "pose", "face"]

def detect_frames(obj):
    if isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
        return obj["frames"]
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (dict, list)):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []

def _get_xyz_from_dict(d):
    x = d.get("x", d.get("X", 0.0))
    y = d.get("y", d.get("Y", 0.0))
    z = d.get("z", d.get("Z", 0.0))
    return float(x), float(y), float(z)

def parse_landmarks(v):
    if v is None:
        return np.zeros((0, 3), dtype=np.float32)

    if isinstance(v, dict):
        if "landmarks" in v:
            return parse_landmarks(v["landmarks"])
        if ("x" in v or "X" in v) and ("y" in v or "Y" in v):
            x, y, z = _get_xyz_from_dict(v)
            return np.array([[x, y, z]], dtype=np.float32)
        return np.zeros((0, 3), dtype=np.float32)

    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        first = v[0]

        if isinstance(first, dict):
            rows = []
            for d in v:
                if isinstance(d, dict) and (("x" in d or "X" in d) and ("y" in d or "Y" in d)):
                    rows.append(_get_xyz_from_dict(d))
            return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)

        if isinstance(first, (list, tuple)) and len(first) >= 2:
            rows = []
            for p in v:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = float(p[0]); y = float(p[1]); z = float(p[2]) if len(p) > 2 else 0.0
                    rows.append((x, y, z))
            return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)

        # ✅ 너 프론트는 flat array(예: 63, 75, 210) 형태라 여기로 들어옴
        if isinstance(first, (int, float)):
            arr = np.array(v, dtype=np.float32)
            if arr.size % 3 == 0:
                return arr.reshape(-1, 3)
            if arr.size % 4 == 0:
                return arr.reshape(-1, 4)[:, :3]
            n = arr.size // 3
            return arr[: n * 3].reshape(-1, 3) if n > 0 else np.zeros((0, 3), dtype=np.float32)

    return np.zeros((0, 3), dtype=np.float32)

def pad_truncate_landmarks(xyz, target_n):
    out = np.zeros((target_n, 3), dtype=np.float32)
    m = min(target_n, xyz.shape[0])
    if m > 0:
        out[:m] = xyz[:m]
    return out

def pad_truncate_frames(frames, target_t):
    if len(frames) == target_t:
        return frames
    if len(frames) > target_t:
        return frames[:target_t]
    last = frames[-1] if frames else {}
    return frames + [last] * (target_t - len(frames))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset", help="dataset root dir (contains label folders)")
    ap.add_argument("--out_dir", default="artifacts", help="output dir")
    args = ap.parse_args()

    DATASET_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Not found: {DATASET_DIR.resolve()}")

    class_dirs = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]
    if not class_names:
        raise RuntimeError(f"No class folders found under {DATASET_DIR}/")

    samples = []
    for cdir in class_dirs:
        for fp in sorted(cdir.glob("*.json")):
            samples.append((cdir.name, fp))
    if not samples:
        raise RuntimeError("No .json files found.")

    print(f"[OK] classes={len(class_names)} samples={len(samples)} from={DATASET_DIR}")

    t_counter = Counter()
    count_counter = {m: Counter() for m in MODALITIES}

    for _, fp in samples:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        frames = detect_frames(obj)
        t_counter[len(frames)] += 1
        for fr in frames[:5]:
            if not isinstance(fr, dict):
                continue
            for m in MODALITIES:
                xyz = parse_landmarks(fr.get(m))
                count_counter[m][xyz.shape[0]] += 1

    T = t_counter.most_common(1)[0][0]
    expected_counts = {}
    for m in MODALITIES:
        nonzero = [(k, v) for k, v in count_counter[m].items() if k > 0]
        expected_counts[m] = sorted(nonzero, key=lambda kv: kv[1], reverse=True)[0][0] if nonzero else 0

    D = sum(expected_counts[m] * 3 for m in MODALITIES)
    if D <= 0:
        raise RuntimeError("Feature dimension D is 0 (parsing failed).")

    print(f"[INFO] T(mode)={T}  T_dist_top={t_counter.most_common(5)}")
    print("[INFO] expected counts:", expected_counts)
    print(f"[INFO] D={D}")

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    X = np.zeros((len(samples), T, D), dtype=np.float32)
    y = np.zeros((len(samples),), dtype=np.int64)

    for i, (label, fp) in enumerate(samples):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        frames = pad_truncate_frames(detect_frames(obj), T)

        seq = np.zeros((T, D), dtype=np.float32)
        for t, fr in enumerate(frames):
            fr = fr if isinstance(fr, dict) else {}
            feats = []
            for m in MODALITIES:
                target_n = expected_counts[m]
                xyz = pad_truncate_landmarks(parse_landmarks(fr.get(m)), target_n)
                feats.append(xyz.reshape(-1))
            vec = np.concatenate(feats, axis=0) if feats else np.zeros((D,), dtype=np.float32)
            seq[t] = vec[:D] if vec.shape[0] >= D else np.pad(vec, (0, D - vec.shape[0]))
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
        # ✅ counts도 저장(훈련/프론트 일치 확인용)
        leftHand_count=np.int32(expected_counts["leftHand"]),
        rightHand_count=np.int32(expected_counts["rightHand"]),
        pose_count=np.int32(expected_counts["pose"]),
        face_count=np.int32(expected_counts["face"]),
        modality_order=np.array(MODALITIES),
    )
    print(f"[DONE] Saved: {out_npz.resolve()}")
    print(f"[SHAPE] X={X.shape} y={y.shape}")

if __name__ == "__main__":
    main()
