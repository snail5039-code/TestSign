import argparse
from pathlib import Path
import json
import numpy as np

def load_label_map(p: Path):
    m = json.loads(p.read_text(encoding="utf-8"))
    # {"WORD00001": 0, ...}
    inv = {int(v): k for k, v in m.items()}
    return m, inv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="ai-server root (dataset_npy_01..16 있는 곳)")
    ap.add_argument("--pattern", default="dataset_npy_*", help="glob pattern for dataset dirs")
    ap.add_argument("--out", default="dataset_npy_all", help="output dir name")
    args = ap.parse_args()

    root = Path(args.root)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    ds_dirs = sorted([p for p in root.glob(args.pattern) if p.is_dir() and (p / "X.npy").exists() and (p / "y.npy").exists() and (p / "label_map.json").exists()])
    if not ds_dirs:
        raise RuntimeError(f"No dataset dirs found under {root} with pattern {args.pattern}")

    # 1) 전체에서 등장하는 label(WORDxxxxx) 수집
    all_labels = set()
    per_dir = []
    for d in ds_dirs:
        lm, inv = load_label_map(d / "label_map.json")
        all_labels.update(lm.keys())
        per_dir.append((d, lm, inv))

    # label 정렬 (WORD00001.. 형태면 문자열 정렬로도 OK)
    all_labels = sorted(all_labels)
    global_map = {lab: i for i, lab in enumerate(all_labels)}
    print(f"[GLOBAL] labels={len(global_map)} (example: {all_labels[:5]})")

    # 2) 데이터 합치기 (로컬 y -> label -> global y)
    X_list = []
    y_list = []
    total = 0

    for d, lm, inv in per_dir:
        X = np.load(d / "X.npy")   # (N,30,822)
        y = np.load(d / "y.npy")   # (N,)
        if X.ndim != 3:
            raise ValueError(f"{d}: X ndim expected 3, got {X.ndim}")
        if X.shape[1:] != (30, 822):
            raise ValueError(f"{d}: X shape expected (*,30,822), got {X.shape}")

        # 로컬 class idx -> label 문자열
        labels = [inv.get(int(c), None) for c in y]
        if any(l is None for l in labels):
            bad = sum(l is None for l in labels)
            raise RuntimeError(f"{d}: {bad} samples have unknown class id in label_map")

        # label -> 글로벌 class idx
        yg = np.array([global_map[l] for l in labels], dtype=np.int64)

        X_list.append(X.astype(np.float32, copy=False))
        y_list.append(yg)

        total += len(yg)
        print(f"[ADD] {d.name}: N={len(yg)}  X={X.shape}  labels_in_dir={len(lm)}  total={total}")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)

    np.save(out / "X.npy", X_all)
    np.save(out / "y.npy", y_all)
    (out / "label_map.json").write_text(json.dumps(global_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[SAVED]")
    print(" out =", out)
    print(" X =", X_all.shape)
    print(" y =", y_all.shape)
    print(" labels =", len(global_map))

if __name__ == "__main__":
    main()
