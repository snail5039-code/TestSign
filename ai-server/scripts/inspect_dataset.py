import json
from pathlib import Path
from collections import Counter, defaultdict

DATASET_DIR = Path("dataset")
EXT = ".json"

def is_landmark_list(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "x" in x[0]

def detect_frames(obj):
    # Case 1: {"frames": [ {...frame...}, {...frame...} ]}
    if isinstance(obj, dict) and "frames" in obj and isinstance(obj["frames"], list):
        return obj["frames"]

    # Case 2: already a list of frames
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        return obj

    # Case 3: single frame dict
    if isinstance(obj, dict):
        return [obj]

    return [obj]

def count_modality(frame, key):
    v = frame.get(key, None)
    if v is None:
        return 0, "missing"

    # hands might be: list of hands (each hand = landmark list)
    if key == "hands":
        if isinstance(v, list) and len(v) > 0:
            # hands = [ [ {x,y,z}, ... ], [ ... ] ] OR hands = [ {x,y,z}, ... ]
            if is_landmark_list(v):
                return len(v), "hands_as_single_landmark_list"
            if isinstance(v[0], list) and len(v[0]) > 0 and is_landmark_list(v[0]):
                return sum(len(hand) for hand in v if is_landmark_list(hand)), "hands_as_list_of_hands"
        return 0, f"hands_unknown_type:{type(v).__name__}"

    # pose/face typically: [ {x,y,z}, ... ]
    if is_landmark_list(v):
        return len(v), "landmark_list"

    return 0, f"unknown_type:{type(v).__name__}"

def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Not found: {DATASET_DIR.resolve()}")

    classes = sorted([p for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError("No class folders under dataset/")

    print(f"[OK] Found {len(classes)} classes:")
    for c in classes:
        print(f"  - {c.name}")

    file_counts = {}
    seq_lens = Counter()
    keys_counter = Counter()

    modality_counts = defaultdict(Counter)  # modality_counts["hands"][21] += 1, etc.
    modality_modes = defaultdict(Counter)   # modality_modes["hands"]["hands_as_list_of_hands"] += 1

    sample_paths = []

    for c in classes:
        files = sorted(c.glob(f"*{EXT}"))
        file_counts[c.name] = len(files)
        sample_paths.extend(files[:3])  # inspect up to 3 per class

        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            frames = detect_frames(obj)
            seq_lens[len(frames)] += 1

            # collect keys from first frame
            if len(frames) > 0 and isinstance(frames[0], dict):
                for k in frames[0].keys():
                    keys_counter[k] += 1

            # collect modality counts per frame
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                for key in ["hands", "pose", "face"]:
                    n, mode = count_modality(fr, key)
                    modality_counts[key][n] += 1
                    modality_modes[key][mode] += 1

    print("\n[INFO] File counts per class:")
    for k, v in file_counts.items():
        print(f"  {k:>8}: {v}")

    print("\n[INFO] Sequence length distribution (frames per json):")
    for k, v in seq_lens.most_common():
        print(f"  len={k}: {v}")

    print("\n[INFO] Top-level keys frequency (from first frame):")
    for k, v in keys_counter.most_common(20):
        print(f"  {k}: {v}")

    print("\n[INFO] Modality landmark count distribution (per frame):")
    for key in ["hands", "pose", "face"]:
        print(f"  - {key}")
        for n, v in modality_counts[key].most_common(10):
            print(f"      count={n}: {v}")
        print(f"    modes: {modality_modes[key].most_common(5)}")

    print("\n[CHECK] Sample files inspected (up to 3 per class):")
    for p in sample_paths:
        print(f"  {p}")

if __name__ == "__main__":
    main()
