# src/debug_relationships.py
import json

with open("data/relationships.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

oppenheimer_triples = [
    t for t in triples
    if "oppenheimer" in t["source"].lower() or "oppenheimer" in t["target"].lower()
]

print(f"Triples mentioning Oppenheimer: {len(oppenheimer_triples)}")
for t in oppenheimer_triples:
    print(f"  {t['source']} --{t['relationship']}--> {t['target']}")