import argparse
from pathlib import Path
import numpy as np


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path.resolve()}")
    return np.load(path, allow_pickle=True)


def l2_mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: (T,D)
    return float(np.mean((a - b) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", default="artifacts/templates.npz", help="templates npz path")
    parser.add_argument("--dataset", default="artifacts/dataset.npz", help="dataset npz path (for testing)")
    parser.add_argument("--index", type=int, default=0, help="sample index from dataset.npz to test")
    args = parser.parse_args()

    tpl = load_npz(Path(args.templates))
    ds = load_npz(Path(args.dataset))

    templates = tpl["templates"].astype(np.float32)     # (C,T,D)
    class_names = tpl["class_names"]                    # (C,)
    X = ds["X"].astype(np.float32)                      # (N,T,D)
    y = ds["y"].astype(np.int64)                        # (N,)

    C, T, D = templates.shape
    N = X.shape[0]

    idx = args.index
    if idx < 0 or idx >= N:
        raise ValueError(f"--index must be in [0, {N-1}]")

    sample = X[idx]  # (T,D)
    true_c = int(y[idx])

    # 각 클래스 템플릿과 거리 계산
    dists = np.array([l2_mean_distance(sample, templates[c]) for c in range(C)], dtype=np.float32)
    pred_c = int(np.argmin(dists))

    # 상위 k개 보기
    topk = 5
    order = np.argsort(dists)[:topk]

    print(f"[INFO] sample index={idx}  X.shape={sample.shape}")
    print(f"[GT]   class={true_c} name={str(class_names[true_c])}")
    print(f"[PRED] class={pred_c} name={str(class_names[pred_c])}  dist={float(dists[pred_c]):.6f}")
    print("\n[TOP5]")
    for rank, c in enumerate(order, 1):
        print(f" {rank:>2}. {str(class_names[int(c)]):<10}  dist={float(dists[int(c)]):.6f}")

    # 전체 정확도(간단 테스트): dataset 전체를 템플릿으로 분류
    correct = 0
    for i in range(N):
        s = X[i]
        d = np.mean((templates - s[None, :, :]) ** 2, axis=(1, 2))  # (C,)
        pc = int(np.argmin(d))
        if pc == int(y[i]):
            correct += 1
    acc = correct / N
    print(f"\n[EVAL] template classifier accuracy on dataset: {acc:.4f} ({correct}/{N})")


if __name__ == "__main__":
    main()
