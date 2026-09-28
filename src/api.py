"""
FastAPI RAG Generation Service (Step 0 Baseline)
Performs vector retrieval via Qdrant and calls a real LLM (Gemini, Anthropic, or OpenAI)
with temperature=0 to produce grounded, cited answers. Fails loudly if no API key is set.
"""

import os
import time
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Embedding model offline mode flags (model weights are cached locally)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

app = FastAPI(
    title="Film Industry RAG Baseline Service",
    description="Retrieval-Augmented Generation API with Real LLM Generation",
    version="1.0.0"
)

# ---------------------------------------------------------------------
# INITIALIZE HEAVY OBJECTS ONCE AT STARTUP
# ---------------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "film_chunks"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

embedder = SentenceTransformer(EMBED_MODEL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


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
    latency_ms: float
    model: str
    generator: str
    input_tokens: int
    output_tokens: int


def call_real_llm(query: str, retrieved_chunks: List[dict]) -> dict:
    """
    Calls Gemini API (gemini-2.5-flash) with temperature=0.0.
    Fails loudly with HTTP 500 if GEMINI_API_KEY is missing or call fails.
    """
    context_blocks = []
    for i, c in enumerate(retrieved_chunks, 1):
        context_blocks.append(f"[Doc {i}: {c['doc_id']}]\n{c['text']}")
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert film industry assistant. Answer the user's question using ONLY the provided context snippets below.\n"
        "For each factual claim, cite the source document title in brackets, e.g. [Oppenheimer (film)].\n"
        "If the answer cannot be found in the context, say 'I don't know based on the provided context.' Do not use external knowledge."
    )

    user_prompt = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{query}\n\n--- ANSWER ---"

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="LLM Generation Error: No API key found. Please set GEMINI_API_KEY in your .env file."
        )

    try:
        from google import genai
        from google.genai import types
        g_client = genai.Client(api_key=gemini_key)
        model_name = "gemini-2.5-flash"
        response = g_client.models.generate_content(
            model=model_name,
            contents=f"{system_prompt}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )
        answer_text = response.text.strip() if response.text else "I don't know based on the provided context."
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if (usage and getattr(usage, "prompt_token_count", None)) else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if (usage and getattr(usage, "candidates_token_count", None)) else 0
        
        return {
            "answer": answer_text,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Error (Gemini): {str(e)}")


@app.get("/health")
def health_check():
    try:
        exists = qdrant_client.collection_exists(COLLECTION_NAME)
        has_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))
        return {
            "status": "ok",
            "collection_exists": exists,
            "collection_name": COLLECTION_NAME,
            "llm_key_configured": has_key
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/ask", response_model=RAGResponse)
@app.post("/ask", response_model=RAGResponse)
def ask_question(query: str = Query(..., description="The factual question to ask"), top_k: int = Query(5, ge=1, le=10)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    # Step 1: Embed Query & Retrieve Chunks
    try:
        q_vec = embedder.encode(query).tolist()
        search_result = qdrant_client.query_points(
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

    # Step 2: Real LLM Generation Call
    llm_res = call_real_llm(query, retrieved_chunks)

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    # Log token usage
    print(f"[RAG API Log] Query: '{query[:40]}...' | Model: {llm_res['model']} | "
          f"In Tokens: {llm_res['input_tokens']} | Out Tokens: {llm_res['output_tokens']} | Latency: {elapsed_ms}ms")

    return RAGResponse(
        query=query,
        answer=llm_res["answer"],
        sources=sorted(list(sources_set)),
        retrieved_chunks=retrieved_chunks,
        latency_ms=elapsed_ms,
        model=llm_res["model"],
        generator="llm",
        input_tokens=llm_res["input_tokens"],
        output_tokens=llm_res["output_tokens"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
