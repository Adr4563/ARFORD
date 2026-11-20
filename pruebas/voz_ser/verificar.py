import json

with open("data/arfjord_unsloth.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(l) for l in f if l.strip()]

print(f"Total: {len(data)}\n")
for i, d in enumerate(data[:8]):
    print(f"{i+1}. {d['prompt']}")
    print(f"   -> {d['completion']}\n")
