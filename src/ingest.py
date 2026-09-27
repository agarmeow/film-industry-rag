"""
Chunk, embed, and ingest the film corpus into Qdrant.
Each chunk's payload carries doc_id (article title) plus any relationship
triples that mention that entity, so GraphRAG can build on this directly.
"""

import os
import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

import re

RAW_DIR = "data/raw"
RELATIONSHIPS_PATH = "data/relationships.json"
COLLECTION_NAME = "film_chunks"
CHUNK_SIZE = 500      # tokens (approximated by words)
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good enough for baseline

embedder = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(host="localhost", port=6333)


def normalize(name):
    name = name.replace(".txt", "")
    return re.sub(r'[^a-z0-9]', '', name.lower())


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def load_relationships_by_entity(path):
    """Index relationships so we can attach relevant triples to each
    document's chunks as metadata."""
    with open(path, "r", encoding="utf-8") as f:
        triples = json.load(f)

    by_entity = {}
    for t in triples:
        by_entity.setdefault(normalize(t["source"]), []).append(t)
        by_entity.setdefault(normalize(t["target"]), []).append(t)
    return by_entity


def setup_collection():
    # 384 = embedding dim for all-MiniLM-L6-v2
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — recreating.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def ingest():
    setup_collection()
    relationships_by_entity = load_relationships_by_entity(RELATIONSHIPS_PATH)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".txt")]
    print(f"Ingesting {len(files)} documents...")

    total_chunks = 0
    batch = []
    BATCH_SIZE = 64

    for fi, fname in enumerate(files, 1):
        doc_id = fname.replace(".txt", "").replace("_", " ")
        path = os.path.join(RAW_DIR, fname)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        if not chunks:
            continue

        related_triples = relationships_by_entity.get(normalize(fname), [])
        vectors = embedder.encode(chunks, show_progress_bar=False)

        for chunk_idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "text": chunk,
                    "relationships": related_triples,  # for later GraphRAG use
                },
            )
            batch.append(point)
            total_chunks += 1

            if len(batch) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                batch = []

        if fi % 20 == 0:
            print(f"  {fi}/{len(files)} documents processed, {total_chunks} chunks so far...")

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"\nDone: {total_chunks} chunks from {len(files)} documents ingested into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    ingest()