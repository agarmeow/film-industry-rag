"""
Prune the corpus down to the intended ~100-150 range by capping
PRODUCED_BY and WON_AWARD per film (STARS was already capped), then
deleting any article files that are no longer referenced.
"""

import json
import os
import re

STUDIO_LIMIT_PER_FILM = 2
AWARD_LIMIT_PER_FILM = 2

with open("data/relationships.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

# --- 1. Cap PRODUCED_BY and WON_AWARD per source film ---
studio_count = {}
award_count = {}
kept = []

for t in triples:
    rel = t["relationship"]
    src = t["source"]

    if rel == "PRODUCED_BY":
        studio_count[src] = studio_count.get(src, 0) + 1
        if studio_count[src] > STUDIO_LIMIT_PER_FILM:
            continue
    elif rel == "WON_AWARD":
        award_count[src] = award_count.get(src, 0) + 1
        if award_count[src] > AWARD_LIMIT_PER_FILM:
            continue

    kept.append(t)

print(f"Relationships: {len(triples)} -> {len(kept)} after capping")

# --- 2. Figure out which article titles are still referenced ---
# (sources and targets in the surviving triples are the entities we want)
needed_titles = set()
for t in kept:
    needed_titles.add(t["source"])
    needed_titles.add(t["target"])

print(f"Entities referenced after capping: {len(needed_titles)}")

# --- 3. Delete article .txt files that are no longer referenced ---
raw_dir = "data/raw"
existing_files = os.listdir(raw_dir)

def normalize(name):
    name = name.replace(".txt", "")
    return re.sub(r'[^a-z0-9]', '', name.lower())

needed_normalized = {normalize(t) for t in needed_titles}

removed = 0
for fname in existing_files:
    if normalize(fname) not in needed_normalized:
        os.remove(os.path.join(raw_dir, fname))
        removed += 1

print(f"Removed {removed} article files no longer referenced")
print(f"Remaining articles: {len(os.listdir(raw_dir))}")

# --- 4. Save the pruned relationships file ---
with open("data/relationships.json", "w", encoding="utf-8") as f:
    json.dump(kept, f, indent=2, ensure_ascii=False)

print("Saved pruned data/relationships.json")