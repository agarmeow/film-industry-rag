# src/debug_retrieve.py
import os

# 1. Does the Oppenheimer article exist at all?
files = os.listdir("data/raw")
matches = [f for f in files if "oppenheimer" in f.lower()]
print("Files matching 'Oppenheimer':", matches)

if matches:
    with open(f"data/raw/{matches[0]}", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"\nArticle length: {len(text.split())} words")
    print(f"First 300 chars:\n{text[:300]}")

    # 2. Does the raw text even mention Cillian Murphy?
    print("\nMentions 'Cillian Murphy':", "Cillian Murphy" in text)
    print("Mentions 'starring':", "starring" in text.lower())
else:
    print("Oppenheimer article was never fetched — check data/relationships.json for it")