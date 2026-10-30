"""사용 예: python examples/client.py [--file examples/request_instruction.json] [--no-selection] [--url http://localhost:8000]"""
import argparse
import json
from pathlib import Path

import httpx

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--file", default=str(Path(__file__).parent / "request_test.json"))
p.add_argument("--no-selection", action="store_true")
args = p.parse_args()

payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
if args.no_selection:
    payload.setdefault("options", {})["use_candidate_selection"] = False

r = httpx.post(f"{args.url}/v1/prompts/generate", json=payload, timeout=600)
r.raise_for_status()
res = r.json()
for c in res.get("candidates") or []:
    mark = "*" if c["selected"] else " "
    print(f"{mark} cand {c['index']}: confidence={c['confidence']}")
print("\n=== initial prompt ===\n" + res["initial_prompt"])
print("\n=== optimized prompt ===\n" + res["optimized_prompt"])
print("\nwarnings:", res["warnings"], "\ntimings(ms):", res["timings_ms"])
