import json
import os
import re

eval_set_path = "data/eval_set.json"
raw_dir = "data/raw"

with open(eval_set_path, "r", encoding="utf-8") as f:
    items = json.load(f)

raw_map = {}
for fname in os.listdir(raw_dir):
    norm = re.sub(r'[^a-z0-9]', '', fname.replace('.txt', '').lower())
    raw_map[norm] = fname

print(f"Total Eval Items: {len(items)}")
print("Spot-checking first 15 questions:\n")

missing_files = 0
grounded_matches = 0

for item in items[:15]:
    norm_doc = re.sub(r'[^a-z0-9]', '', item['source_doc'].lower())
    fname = raw_map.get(norm_doc)
    if not fname:
        print(f"[{item['id']:02d}] MISSING FILE: {item['source_doc']}")
        missing_files += 1
        continue
    
    text = open(os.path.join(raw_dir, fname), encoding="utf-8").read()
    ref = item['reference_answer']
    is_grounded = ref.lower() in text.lower()
    if is_grounded:
        grounded_matches += 1
    
    print(f"[{item['id']:02d}] File: {fname}")
    print(f"     Q: {item['question']}")
    print(f"     Ref: {ref} | Grounded in raw text: {is_grounded}\n")

print(f"Spot Check Results: {grounded_matches}/15 grounded directly in raw Wikipedia text.")
