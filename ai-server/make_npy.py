import argparse
from pathlib import Path
import time
import json
import re
import numpy as np
from tqdm import tqdm

from dataset import parse_one_json   # 한 json(=1프레임) -> (1,411)
from feature import add_delta        # (30,411)->(30,822)

WORD_RE = re.compile(r"NIA_SL_WORD(\d+)_", re.IGNORECASE)

def extract_word_label(seq_dir: Path):
    m = WORD_RE.search(seq_dir.name)
    if not m:
        return None
    num = int(m.group(1))
    return f"WORD{num:05d}"  # WORD00001

def resample_or_pad(seq_411: np.ndarray, target_frames: int):
    T = seq_411.shape[0]
    if T <= 0:
        return np.zeros((target_frames, 411), dtype=np.float32)
    if T == target_frames:
        return seq_411.astype(np.float32)
    if T > target_frames:
        idx = np.linspace(0, T - 1, target_frames).round().astype(int)
        return seq_411[idx].astype(np.float32)
    pad = np.zeros((target_frames - T, 411), dtype=np.float32)
    return np.concatenate([seq_411.astype(np.float32), pad], axis=0)

def build_sequence_from_dir(seq_dir: Path):
    files = sorted(seq_dir.glob("*_keypoints.json"))
    if not files:
        raise ValueError("no *_keypoints.json")
    feats = []
    for fp in files:
        f = parse_one_json(fp)  # (1,411)
        feats.append(f[0])
    return np.stack(feats, axis=0).astype(np.float32)  # (T,411)

def iter_sequence_dirs_fast(src: Path):
    """
    src는 보통 ...\\01_real_word_keypoint\\01 같은 곳(거기 아래에 NIA_SL_WORD... 폴더가 많음)
    여기서 NIA_SL_WORD* 폴더만 빠르게 나열 (rglob 금지)
    """
    # src 자체가 시퀀스 폴더인 경우
    if any(src.glob("*_keypoints.json")) and src.is_dir():
        yield src
        return

    # 1단계 하위 디렉토리 중 NIA_SL_WORD...만 탐색
    # (src 아래가 바로 NIA_SL_WORD... 폴더들이라면 이게 가장 빠름)
    for p in sorted(src.iterdir()):
        if p.is_dir() and p.name.upper().startswith("NIA_SL_WORD"):
            if any(p.glob("*_keypoints.json")):
                yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="시퀀스 폴더들이 있는 위치(예: ...\\01_real_word_keypoint\\01)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=30)         # ✅ AIHub 30
    ap.add_argument("--max_labels", type=int, default=30)     # ✅ 앞에서부터 30개 라벨만
    ap.add_argument("--max_per_label", type=int, default=5)   # ✅ 라벨당 몇 개만(빠르게 테스트)
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label_map = {}           # label -> class idx
    counts = {}              # label -> collected samples
    X_list, y_list = [], []

    total_ok, total_bad = 0, 0
    t0 = time.time()

    # 진행 로그
    print(f"[SCAN] src={src}")
    print(f"[CONF] max_labels={args.max_labels}, max_per_label={args.max_per_label}, frames={args.frames}")

    pbar = tqdm(iter_sequence_dirs_fast(src), desc="scan-seq-dirs")
    for seq_dir in pbar:
        label = extract_word_label(seq_dir)
        if label is None:
            total_bad += 1
            continue

        # 라벨 30개 꽉 찼으면, 새로운 라벨은 스킵
        if label not in label_map:
            if len(label_map) >= args.max_labels:
                # ✅ 여기서 바로 끊어버림(앞에서부터 30개만)
                break
            label_map[label] = len(label_map)
            counts[label] = 0

        # 라벨당 max_per_label 채우면 해당 라벨은 더 안 받음
        if counts[label] >= args.max_per_label:
            continue

        try:
            seq = build_sequence_from_dir(seq_dir)            # (T,411)
            T = int(seq.shape[0])
            seq30 = resample_or_pad(seq, args.frames)         # (30,411)
            seq822 = add_delta(seq30)                         # (30,822)

            X_list.append(seq822)
            y_list.append(label_map[label])
            counts[label] += 1

            total_ok += 1
            if total_ok % args.log_every == 0:
                pbar.set_postfix({
                    "labels": len(label_map),
                    "ok": total_ok,
                    "bad": total_bad,
                    "last": label,
                    "frames": T
                })

        except Exception as e:
            total_bad += 1
            if total_bad <= 10:
                tqdm.write(f"[SKIP] {seq_dir.name} reason={type(e).__name__}: {e}")

        # ✅ 모든 라벨이 max_per_label까지 채워지면 끝
        if len(label_map) >= args.max_labels and all(counts[l] >= args.max_per_label for l in label_map.keys()):
            break

    if len(X_list) == 0:
        raise RuntimeError("No samples created. Check src path / json parsing.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N,30,822)
    y = np.array(y_list, dtype=np.int64)

    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    (out / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    dt = time.time() - t0
    print("\n[SAVED]")
    print(" out =", out)
    print(" X =", X.shape, "y =", y.shape)
    print(" labels =", len(label_map), "per_label_target =", args.max_per_label)
    print(f" ok={total_ok} bad={total_bad} time={dt:.1f}s")

if __name__ == "__main__":
    main()
