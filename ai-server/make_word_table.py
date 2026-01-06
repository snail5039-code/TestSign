import argparse
from pathlib import Path
import json
import re

WORD_RE = re.compile(r"NIA_SL_WORD(\d+)_", re.IGNORECASE)
HANGUL_RE = re.compile(r"[가-힣]")

def iter_strings(obj, path="$", depth=0, max_depth=8):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_strings(v, f"{path}.{k}", depth + 1, max_depth)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_strings(v, f"{path}[{i}]", depth + 1, max_depth)
    elif isinstance(obj, str):
        s = obj.strip()
        if s:
            yield (path, s)

def score_candidate(s: str) -> int:
    """
    '단어' 같은 한글 문자열을 고르기 위한 점수 함수.
    점수 높을수록 채택.
    """
    if not HANGUL_RE.search(s):
        return -10_000

    score = 0
    t = s.strip()

    # 너무 긴 문장/설명은 패널티
    L = len(t)
    if L <= 2:
        score += 60
    elif L <= 6:
        score += 100
    elif L <= 12:
        score += 60
    elif L <= 20:
        score += 10
    else:
        score -= 200

    # 공백/문장부호 있으면 단어 가능성 낮음
    if " " in t:
        score -= 80
    if any(ch in t for ch in [".", ",", "?", "!", ":", ";", "/", "\\", "_"]):
        score -= 30

    # 숫자/영문 섞이면 패널티
    if re.search(r"[A-Za-z0-9]", t):
        score -= 50

    # 흔히 메타에 들어가는 값 패널티
    bad_tokens = ["http", "NIA", "REAL", "keypoints", "morpheme", "label", "version"]
    if any(bt.lower() in t.lower() for bt in bad_tokens):
        score -= 200

    return score

def guess_korean_word(obj: dict):
    cands = []
    for path, s in iter_strings(obj):
        if HANGUL_RE.search(s):
            cands.append((score_candidate(s), path, s))

    if not cands:
        return None, None, None

    cands.sort(key=lambda x: x[0], reverse=True)
    best = cands[0]
    return best[2], best[1], best[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"morpheme 루트 (예: ...\01_real_word_morpheme\morpheme)")
    ap.add_argument("--out", default="word_table.json")
    ap.add_argument("--max_words", type=int, default=30)
    ap.add_argument("--debug", action="store_true", help="선택된 값 경로/점수 출력")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    mapping = {}

    # 01~16 순서대로 돌면서 WORD00001~max_words 채워질 때까지만
    for i in range(1, 17):
        d = root / f"{i:02d}"
        if not d.exists():
            continue

        files = sorted(d.glob("NIA_SL_WORD*_morpheme.json"))
        for fp in files:
            m = WORD_RE.search(fp.name)
            if not m:
                continue
            num = int(m.group(1))
            if num < 1 or num > args.max_words:
                continue

            label = f"WORD{num:05d}"
            if label in mapping:
                continue

            obj = json.loads(fp.read_text(encoding="utf-8"))
            word, where, sc = guess_korean_word(obj)

            if word is None:
                # 못 찾으면 일단 라벨로 fallback (하지만 debug로 확인 가능)
                word = label
                where = None
                sc = None

            mapping[label] = word

            if args.debug:
                print(f"{label} <= {fp.name}")
                print(f"  pick: {word}")
                if where is not None:
                    print(f"  path: {where}  score: {sc}")

            if len(mapping) >= args.max_words:
                break

        if len(mapping) >= args.max_words:
            break

    out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SAVED]", out)
    print("labels:", len(mapping))
    for k in list(mapping.keys())[:10]:
        print(k, "=>", mapping[k])

if __name__ == "__main__":
    main()
