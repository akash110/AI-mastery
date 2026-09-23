# Module 1: Retrieval-Augmented Generation (RAG) Architecture

## 1. The Enterprise Problem
LLMs are frozen at training time and have no access to your private,
constantly-changing company data (runbooks, wikis, ticket history, HCON
configs). Asked about internal systems, they either say "I don't know" or —
worse — **hallucinate** a plausible-sounding but wrong answer. You cannot
ship that to customers or oncall engineers.

## 2. The Engineering Solution
Build a pipeline that retrieves the *right* private text and hands it to the
LLM as context at query time, so the model answers from facts instead of
memory.

### 2.1 Data Ingestion & Chunking
- **Fixed-size chunking**: split by character/token count (e.g. 500 chars).
  Simple, fast, but can cut a sentence or table in half.
- **Semantic chunking**: split on natural boundaries (paragraphs, headers,
  markdown sections) so each chunk is a coherent idea. Better retrieval
  precision, costs more preprocessing.
- **Chunk overlap** (10–20%): each chunk repeats the tail of the previous
  chunk. Prevents an answer that straddles a chunk boundary from being lost
  entirely from either chunk.

  Analogy to your world: this is like choosing a partition/window size for a
  batch ETL job — too small and you lose context across the boundary, too
  large and each unit becomes noisy/expensive to process.

### 2.2 Vectorization & Embeddings
- An **embedding model** (OpenAI `text-embedding-3-*`, or local open-source
  ones like `nomic-embed-text`, `bge-small`, `all-MiniLM-L6-v2`) converts
  text into a fixed-length numeric vector where semantically similar text
  ends up close together in vector space.
- A **vector database** (Pinecone, Milvus, ChromaDB, pgvector) stores those
  vectors and supports fast nearest-neighbor search at scale (millions of
  vectors, sub-second lookup via approximate nearest-neighbor indexes like
  HNSW).

### 2.3 The Retrieval Step
- **Cosine similarity**: measures the angle between two vectors (not their
  magnitude) — 1.0 means identical direction (very similar meaning), 0 means
  unrelated, -1 means opposite. Vector search returns the top-K chunks by
  this score.
- **Hybrid search**: pure vector search misses exact keyword/ID matches
  (e.g. an error code "ERR_504_TIMEOUT") because embeddings capture *meaning*
  not exact tokens. **BM25** is a classic keyword-frequency ranking
  algorithm. Hybrid search runs both and merges/re-weights the results —
  best of both worlds.
- **Re-ranking**: vector search over millions of chunks is optimized for
  speed, not precision, so it over-fetches (e.g. top 20–50 candidates). A
  slower but more accurate **cross-encoder** model then re-scores just those
  candidates by jointly encoding (query, chunk) pairs, and only the true top
  3–5 go to the LLM. This is the single highest-leverage step for cutting
  hallucination from bad retrieval.

## 3. The Evaluation Metric
- **Context Relevance** (did we retrieve the right chunks?) — covered in
  depth in Module 3, but it starts here: if chunking/retrieval is bad, no
  amount of prompt engineering downstream fixes it.
- Practical dev-time check: log retrieved chunks per query and manually (or
  with an LLM judge) verify they actually contain the answer before blaming
  the generation step.

## Run it
```bash
python module1_rag/rag_pipeline.py
```
This ingests a handful of fake "runbook" documents, chunks them, embeds them
locally with Ollama's `nomic-embed-text`, stores them in ChromaDB, then runs
hybrid (BM25 + vector) search with a simple re-ranking step for a sample
query.
