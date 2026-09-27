# src/test_retrieve.py
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

def retrieve(query, top_k=5):
    q_vec = embedder.encode(query).tolist()
    return client.query_points(
        collection_name="film_chunks",
        query=q_vec,
        limit=top_k,
    ).points

if __name__ == "__main__":
    queries = [
        "Who starred in Oppenheimer?",
        "Which films directed by Christopher Nolan won awards?",
    ]

    for q in queries:
        print(f"\n==========================================")
        print(f"QUERY: {q}")
        print(f"==========================================")
        results = retrieve(q)
        for r in results:
            doc_id = r.payload['doc_id']
            snippet = r.payload['text'][:120].replace('\n', ' ')
            rel_count = len(r.payload.get('relationships', []))
            print(f"[{r.score:.3f}] {doc_id} (Triples: {rel_count})")
            print(f"       Snippet: {snippet}...")
            if r.payload.get('relationships'):
                sample_rel = r.payload['relationships'][:2]
                print(f"       Sample Metadata: {sample_rel}")
            print()