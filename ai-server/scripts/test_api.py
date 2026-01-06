import json, sys
import requests

FILE = sys.argv[1] if len(sys.argv) > 1 else r"dataset/eat/eat_000001.json"
URL = "http://127.0.0.1:8000/predict"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

payload = {"frames": data["frames"]}
r = requests.post(URL, json=payload, timeout=30)
r.raise_for_status()
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
