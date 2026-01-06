import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_npz", default="artifacts/dataset.npz", help="input dataset npz path")
    parser.add_argument("--out_npz", default="artifacts/templates.npz", help="output templates npz path")
    args = parser.parse_args()

    in_path = Path(args.in_npz)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path.resolve()}")

    data = np.load(in_path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise ValueError("dataset.npz must contain keys: 'X' and 'y'")

    X = data["X"].astype(np.float32)  # (N,T,D)
    y = data["y"].astype(np.int64)    # (N,)

    # class_names는 문자열 배열일 수도, object 배열일 수도 있어 allow_pickle 사용
    class_names = data["class_names"] if "class_names" in data else None

    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,D). got shape={X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be 1D with same length as X. got y={y.shape}, X={X.shape}")

    N, T, D = X.shape
    num_classes = int(y.max()) + 1

    # class_names가 없으면 y 기반으로 임시 생성
    if class_names is None:
        class_names = np.array([f"class_{i}" for i in range(num_classes)], dtype=object)

    if len(class_names) != num_classes:
        # 혹시 class_names 길이가 다르면 num_classes에 맞춰 보정
        # (정상이라면 이 상황은 없어야 함)
        class_names = np.array([str(c) for c in class_names], dtype=object)
        if len(class_names) < num_classes:
            extra = [f"class_{i}" for i in range(len(class_names), num_classes)]
            class_names = np.concatenate([class_names, np.array(extra, dtype=object)], axis=0)
        else:
            class_names = class_names[:num_classes]

    templates = np.zeros((num_classes, T, D), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        counts[c] = len(idx)
        if len(idx) == 0:
            # 해당 클래스 샘플이 없으면 0 템플릿 유지
            continue
        templates[c] = X[idx].mean(axis=0)

    np.savez_compressed(
        out_path,
        templates=templates,         # (C,T,D)
        class_names=class_names,     # (C,)
        counts=counts,               # (C,)
        T=np.array(T, dtype=np.int64),
        D=np.array(D, dtype=np.int64),
    )

    print(f"[OK] Loaded: {in_path.resolve()}")
    print(f"[OK] Saved:  {out_path.resolve()}")
    print(f"[INFO] X={X.shape} y={y.shape} classes={num_classes}")
    print("[INFO] counts per class:")
    for c in range(num_classes):
        print(f"  - {str(class_names[c])}: {int(counts[c])}")
    print(f"[SHAPE] templates={templates.shape} (C={num_classes}, T={T}, D={D})")


if __name__ == "__main__":
    main()
