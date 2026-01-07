# scripts/make_label_map_from_json.py
import argparse, json
from pathlib import Path

CAND_KEYS = ["label_ko", "labelKo", "korean", "ko_label", "koreanLabel"]

def extract_ko(obj: dict):
    # top-level
    for k in CAND_KEYS:
        if isinstance(obj.get(k), str) and obj.get(k).strip():
            return obj[k].strip()

    # meta 안에 들어있는 경우
    meta = obj.get("meta")
    if isinstance(meta, dict):
        for k in CAND_KEYS:
            if isinstance(meta.get(k), str) and meta.get(k).strip():
                return meta[k].strip()

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dataset root (contains class folders)")
    ap.add_argument("--out", required=True, help="output labels_ko.json path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    mapping = {}
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    for cdir in class_dirs:
        label = cdir.name  # 학습 클래스 키(폴더명)
        ko = None

        # 해당 클래스 폴더에서 아무 json 하나만 읽어서 ko 라벨 뽑기
        for fp in sorted(cdir.glob("*.json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            ko = extract_ko(obj)
            if ko:
                break

        mapping[label] = ko if ko else label  # 없으면 폴더명 fallback

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] saved: {out_path.resolve()}")
    print("[MAP]", mapping)

if __name__ == "__main__":
    main()
