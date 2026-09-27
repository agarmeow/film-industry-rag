"""
FastAPI RAG Service — Retrieval-Augmented Generation Endpoint.
Retrieves top-k context chunks from Qdrant, builds a grounded prompt,
generates an answer via Gemini API (or fallback extractor), and returns
the answer with document citations.
"""

import os
import ssl
import json
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Suppress SSL verification warnings if local proxy/cert issues exist
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

app = FastAPI(
    title="Film Industry RAG Baseline Service",
    description="Retrieval-Augmented Generation API for Film Industry Wikipedia Corpus",
    version="1.0.0"
)

# Initialize Qdrant Client & Embedding Model
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "film_chunks"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

embedder = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class ChunkPayload(BaseModel):
    doc_id: str
    chunk_index: int
    score: float
    text: str
    relationships: List[dict] = []


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    retrieved_chunks: List[ChunkPayload]


def generate_answer_from_context(query: str, chunks: List[dict]) -> str:
    """
    Generates a grounded answer using Gemini API if GEMINI_API_KEY is set,
    or falls back to an intelligent context-grounded synthesis engine.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    context_text = "\n\n".join(
        f"[Source: {c['doc_id']}]\n{c['text']}" for c in chunks
    )

    if api_key:
        try:
            from google import genai
            genai_client = genai.Client(api_key=api_key)
            prompt = (
                f"You are an expert film industry assistant. Answer the user's question accurately using ONLY the context provided below.\n"
                f"If the answer cannot be found in the context, state that clearly. Include citations to source document titles.\n\n"
                f"--- CONTEXT ---\n{context_text}\n\n"
                f"--- QUESTION ---\n{query}"
            )
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",  # or gemini-flash-latest
                contents=prompt
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini API call failed, using fallback synthesis: {e}")

    # Fallback context-grounded synthesis
    # Look for direct factual matches in chunks & relationship metadata
    query_lower = query.lower()
    relevant_sentences = []
    
    for c in chunks:
        doc = c['doc_id']
        text = c['text']
        rels = c.get('relationships', [])

        # Check relationships metadata for direct graph matches
        for rel in rels:
            src = rel.get('source', '')
            target = rel.get('target', '')
            r_type = rel.get('relationship', '')
            if r_type == 'DIRECTED' and ('who directed' in query_lower or 'director' in query_lower):
                if src.lower() in query_lower or target.lower() in query_lower:
                    return f"{src} directed {target}."
            elif r_type == 'STARS' and ('who starred' in query_lower or 'star' in query_lower or 'actor' in query_lower or 'cast' in query_lower):
                if src.lower() in query_lower:
                    stars = [r['target'] for r in rels if r.get('relationship') == 'STARS']
                    if stars:
                        return f"{src} stars {', '.join(set(stars))}."
            elif r_type == 'WON_AWARD' and ('award' in query_lower or 'win' in query_lower or 'won' in query_lower):
                if src.lower() in query_lower or target.lower() in query_lower:
                    awards = [r['target'] for r in rels if r.get('relationship') == 'WON_AWARD']
                    if awards:
                        return f"{src} won the following award(s): {', '.join(set(awards))}."

        # Extract matching sentence from text
        sentences = text.split('. ')
        for s in sentences:
            if any(term in s.lower() for term in query_lower.replace('?', '').split() if len(term) > 3):
                relevant_sentences.append((s.strip(), doc))

    if relevant_sentences:
        top_sentence, top_doc = relevant_sentences[0]
        if not top_sentence.endswith('.'):
            top_sentence += '.'
        return f"{top_sentence} (Source: {top_doc})"

    top_doc = chunks[0]['doc_id'] if chunks else "Unknown"
    return f"Based on {top_doc}, {chunks[0]['text'][:200]}..."


@app.get("/health")
def health_check():
    try:
        exists = client.collection_exists(COLLECTION_NAME)
        return {"status": "ok", "collection_exists": exists, "collection_name": COLLECTION_NAME}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/ask", response_model=RAGResponse)
@app.post("/ask", response_model=RAGResponse)
def ask_question(query: str = Query(..., description="The factual question to ask"), top_k: int = Query(5, ge=1, le=10)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        q_vec = embedder.encode(query).tolist()
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=top_k,
        ).points
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant query failed: {str(e)}")

    retrieved_chunks = []
    sources_set = set()

    for p in search_result:
        doc_id = p.payload.get("doc_id", "Unknown")
        sources_set.add(doc_id)
        retrieved_chunks.append({
            "doc_id": doc_id,
            "chunk_index": p.payload.get("chunk_index", 0),
            "score": round(p.score, 4),
            "text": p.payload.get("text", ""),
            "relationships": p.payload.get("relationships", []),
        })

    answer = generate_answer_from_context(query, retrieved_chunks)

    return RAGResponse(
        query=query,
        answer=answer,
        sources=sorted(list(sources_set)),
        retrieved_chunks=retrieved_chunks
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
