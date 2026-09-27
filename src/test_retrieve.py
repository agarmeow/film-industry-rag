# src/test_retrieve.py
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
    results = retrieve("Who starred in Oppenheimer?")
    for r in results:
        print(f"[{r.score:.3f}] {r.payload['doc_id']} — {r.payload['text'][:100]}...")