# src/debug_retrieve2.py
import os

files = os.listdir("data/raw")
matches = [f for f in files if "oppenheimer" in f.lower()]
print("Files matching 'Oppenheimer':", matches)

if matches:
    with open(f"data/raw/{matches[0]}", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Article length: {len(text.split())} words")
    print("Mentions 'Cillian Murphy':", "Cillian Murphy" in text)

    # Show where in the article Cillian Murphy is mentioned, and what
    # ~500-word chunk window that would land in
    idx = text.find("Cillian Murphy")
    if idx != -1:
        words_before = text[:idx].split()
        print(f"'Cillian Murphy' appears after word #{len(words_before)}")
        # crude estimate of which chunk (0-indexed, 500 words, 50 overlap)
        chunk_num = len(words_before) // (500 - 50)
        print(f"Roughly falls in chunk #{chunk_num}")
else:
    print("Still missing — something else is wrong")