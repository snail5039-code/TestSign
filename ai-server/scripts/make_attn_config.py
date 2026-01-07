import os
import json
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="artifacts/dataset.npz")
    ap.add_argument("--out_dir", required=True, help="artifacts folder (same as training out_dir)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--zero_face", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    if X.ndim != 3:
        raise RuntimeError(f"X must be (N,T,D), got {X.shape}")

    N, T, D = X.shape

    # counts (if build_dataset.py saved them). Otherwise fall back to common defaults.
    left_cnt  = int(data["leftHand_count"])  if "leftHand_count"  in data.files else 21
    right_cnt = int(data["rightHand_count"]) if "rightHand_count" in data.files else 21
    pose_cnt  = int(data["pose_count"])      if "pose_count"      in data.files else 25
    face_cnt  = int(data["face_count"])      if "face_count"      in data.files else 70

    cfg = {
        "T": int(T),
        "input_dim": int(D),
        "hidden": int(args.hidden),
        "layers": int(args.layers),
        "dropout": float(args.dropout),
        "zero_face": bool(args.zero_face),
        "modality_order": ["leftHand", "rightHand", "pose", "face"],
        "counts": {
            "leftHand": int(left_cnt),
            "rightHand": int(right_cnt),
            "pose": int(pose_cnt),
            "face": int(face_cnt),
        },
    }

    out_path = os.path.join(args.out_dir, "attn_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[SAVED]", out_path)
    print("[INFO] T,D =", T, D)
    print("[INFO] counts =", cfg["counts"])

if __name__ == "__main__":
    main()
