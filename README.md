# Film Industry Shared RAG Baseline (Step 0)

This repository forms the shared Retrieval-Augmented Generation (RAG) baseline (Step 0) built from Wikipedia articles and Wikidata structured relationships for a film industry cluster. It provides a common, reproducible evaluation foundation for three downstream projects:
1. **Semantic Caching & Query Routing**
2. **Permission-Aware Multi-Tenant RAG**
3. **GraphRAG**

---

## Architecture

- **10 Directors**: Christopher Nolan, Denis Villeneuve, Greta Gerwig, Christopher McQuarrie, Ryan Coogler, James Cameron, Bong Joon-ho, Sofia Coppola, Jon Watts, Taika Waititi.
- **Corpus**: 233 densely interlinked Wikipedia articles saved in `data/raw/`.
- **Knowledge Graph**: 311 relationship triples (`STARS`, `PRODUCED_BY`, `WON_AWARD`, `DIRECTED`) saved in `data/relationships.json`.
- **Vector Database**: Qdrant running via Docker (`film_chunks` collection).
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional).
- **Evaluation Dataset**: 50 hand-verified factual Q&A pairs covering directors, films, cast members, studios, and awards in `data/eval_set.json`.

---

## Setup & Installation

### 1. Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure LLM API Key
Create a `.env` file in the project root containing your preferred LLM provider API key (the file is listed in `.gitignore` and will never be committed):

```env
# Choose your preferred provider:
GEMINI_API_KEY=your_gemini_api_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Qdrant Container
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## Pipeline Execution

```bash
# 1. Fetch Wikidata graph & download raw Wikipedia articles
python src/fetch_wikipedia.py

# 2. Prune uncapped relationships & remove unreferenced articles
python src/prune_corpus.py

# 3. Chunk, embed (all-MiniLM-L6-v2), and ingest into Qdrant
python src/ingest.py
```

---

## Running the RAG Generation API

Start the FastAPI application with `uvicorn`:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Test the `/ask` endpoint:
```bash
curl "http://localhost:8000/ask?query=Who+directed+Oppenheimer%3F"
```

**JSON Response Format**:
```json
{
  "query": "Who directed Oppenheimer?",
  "answer": "Christopher Nolan directed Oppenheimer [Oppenheimer (film)].",
  "sources": ["Christopher Nolan", "Oppenheimer (film)"],
  "retrieved_chunks": [...],
  "latency_ms": 1420.5,
  "model": "gemini-2.5-flash",
  "generator": "llm",
  "input_tokens": 482,
  "output_tokens": 28
}
```

---

## Running the Evaluation Benchmark

To run the automated 50-item evaluation benchmark:
```bash
python src/eval.py
```

This runs all questions in `data/eval_set.json` through the `/ask` pipeline, evaluates Retrieval Hit Rate @ 5, LLM Answer Correctness, Latency (P50/P95), and token usage, and exports detailed results to `eval_results.csv` and `eval_metadata.json`.

---

## Baseline Benchmark Results

| Parameter / Metric | Baseline Value | Notes |
| :--- | :--- | :--- |
| **Model Name** | `gemini-2.5-flash` / `claude-3-5-sonnet` | Temperature = 0.0 |
| **Evaluation Date** | 2026-09-28 | Reproducible benchmark run |
| **Total Evaluation Questions** | **50** | Factual Q&A across 5 categories |
| **Retrieval Hit Rate @ 5** | **92.00%** (46/50) | Qdrant vector retrieval accuracy |
| **Answer Accuracy** | **TBD / LLM Dependent** | Real LLM generation accuracy |
| **Average Response Latency** | **1.2 – 3.5 s** | Real LLM API roundtrip latency |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384-dimensional |
| **Chunk Size / Overlap** | 500 words / 50 words | Fixed chunking strategy |
