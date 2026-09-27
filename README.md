# Film Industry Shared RAG Baseline (Step 0)

This repository serves as the shared RAG baseline (Step 0) built from Wikipedia and Wikidata. It forms the common foundation for three downstream projects:
1. **Semantic Caching**
2. **Permission-aware Multi-tenant RAG (ACL demo)**
3. **GraphRAG**

## Corpus Architecture

- **10 Directors**: Christopher Nolan, Denis Villeneuve, Greta Gerwig, Christopher McQuarrie, Ryan Coogler, James Cameron, Bong Joon-ho, Sofia Coppola, Jon Watts, Taika Waititi.
- **Interconnected Graph**: Films, Cast Members, Production Studios, and Awards.
- **Wikidata Ingestion**: SPARQL queries using property relationships (`P57` director, `P161` cast, `P272` studio, `P166` award) batched and ranked by notability (`wikibase:sitelinks`).
- **Disambiguated Article Alignment**: Wikipedia article titles (e.g. `Oppenheimer (film)`) strictly map to raw text files and relationship triples.

## Setup & Ingestion

### 1. Requirements & Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Qdrant Container
```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 3. Pipeline Steps
```bash
# Fetch Wikidata graph & download raw Wikipedia articles
python src/fetch_wikipedia.py

# Prune uncapped relationships & non-referenced articles
python src/prune_corpus.py

# Chunk, embed (all-MiniLM-L6-v2), and ingest into Qdrant
python src/ingest.py

# Test vector search retrieval
python src/test_retrieve.py
```

## Retrieval Diagnostics

Test retrieval verifies semantic retrieval accuracy:
```bash
python src/test_retrieve.py
```
Output returns top-k matching chunks with similarity scores and attached metadata triples.
