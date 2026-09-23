"""
Module 1: RAG Architecture — runnable demo.

Pipeline stages (each maps to a NOTES.md section):
  1. Ingest + chunk documents (fixed-size, with overlap)
  2. Embed chunks locally via Ollama (nomic-embed-text)
  3. Store/retrieve via ChromaDB (vector search, cosine similarity)
  4. Hybrid search: merge BM25 (keyword) + vector search results
  5. Re-rank the merged candidates with a lightweight cross-encoder
  6. Feed the top chunks to a local LLM (llama3.2) to answer

Requires: `ollama serve` running, with `ollama pull nomic-embed-text` and
`ollama pull llama3.2` already done. See ../README.md for setup.
"""

import ollama
import chromadb
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

# --- Fake "runbook" corpus: stand-in for internal wiki/Confluence pages ---
DOCUMENTS = [
    {
        "id": "runbook_1",
        "text": (
            "Incident Runbook: QuickETL Pipeline Timeout.\n"
            "If a QuickETL HCON job fails with ERR_504_TIMEOUT, first check "
            "whether the source Datalake bucket has FGAC (Fine-Grained "
            "Access Control) enabled. FGAC-enabled buckets require the "
            "SparkLoaderStep operator to use the 'assumeRole' parameter, "
            "otherwise reads silently time out after 300 seconds. "
            "Fix: add assumeRole=datalake-reader-role to the operator config "
            "and re-run the job."
        ),
    },
    {
        "id": "runbook_2",
        "text": (
            "Incident Runbook: Vector Database High Latency.\n"
            "If ChromaDB or Pinecone query latency exceeds 500ms p99, check "
            "index fragmentation first. Rebuilding the HNSW index after "
            "large batch inserts typically resolves it. Also verify the "
            "embedding dimension matches the collection's configured "
            "dimension — a mismatch causes silent full-table scans instead "
            "of using the index."
        ),
    },
    {
        "id": "runbook_3",
        "text": (
            "Incident Runbook: LLM Agent Stuck in Tool-Call Loop.\n"
            "If the oncall agent repeatedly calls the same diagnostic tool "
            "without terminating, check the ReAct loop's max_iterations "
            "guardrail. It should hard-stop at 5 iterations and escalate to "
            "a human. Also verify the tool's return value is not empty — "
            "an empty observation can cause the agent to retry indefinitely."
        ),
    },
    {
        "id": "runbook_4",
        "text": (
            "Incident Runbook: PII Leak in Support Ticket Export.\n"
            "If a data export job includes unmasked credit card numbers or "
            "SSNs, the PII redaction step in the ETL pipeline was likely "
            "skipped. All exports must pass through the redact_pii operator "
            "before landing in the shared analytics bucket. Escalate to the "
            "data governance team immediately if unmasked PII is found in "
            "an already-shared location."
        ),
    },
]


def chunk_text(text: str, chunk_size: int = 220, overlap_ratio: float = 0.15):
    """Fixed-size chunking with overlap (Module 1, section 2.1)."""
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step back by the overlap amount
    return chunks


def build_corpus():
    """Chunk every document, keeping track of source doc id."""
    chunk_records = []
    for doc in DOCUMENTS:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunk_records.append(
                {"chunk_id": f"{doc['id']}_chunk{i}", "source": doc["id"], "text": chunk}
            )
    return chunk_records


def embed_and_store(chunk_records):
    """Embed each chunk locally and store in an in-memory ChromaDB collection."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("runbooks")

    for rec in chunk_records:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=rec["text"])["embedding"]
        collection.add(
            ids=[rec["chunk_id"]],
            embeddings=[embedding],
            documents=[rec["text"]],
            metadatas=[{"source": rec["source"]}],
        )
    return collection


def vector_search(collection, query: str, top_k: int = 5):
    """Cosine-similarity search via ChromaDB (Module 1, section 2.3)."""
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return list(zip(results["ids"][0], results["documents"][0]))


def bm25_search(chunk_records, query: str, top_k: int = 5):
    """Keyword search via BM25 — catches exact terms embeddings can miss."""
    tokenized_corpus = [rec["text"].lower().split() for rec in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunk_records, scores), key=lambda x: x[1], reverse=True)
    return [(rec["chunk_id"], rec["text"]) for rec, score in ranked[:top_k] if score > 0]


def hybrid_search(collection, chunk_records, query: str, top_k: int = 5):
    """Merge vector + BM25 results, deduplicating by chunk id (Module 1, 2.3)."""
    vec_results = vector_search(collection, query, top_k)
    kw_results = bm25_search(chunk_records, query, top_k)

    merged = {}
    for chunk_id, text in vec_results + kw_results:
        merged[chunk_id] = text  # dedupe; a real system would blend scores (e.g. RRF)
    return list(merged.items())


def rerank(query: str, candidates: list, top_n: int = 3):
    """
    Lightweight re-ranking stand-in for a cross-encoder (Module 1, 2.3).
    A production system would use sentence-transformers' CrossEncoder
    (e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2') to jointly score
    (query, chunk) pairs. Here we approximate with simple term-overlap
    scoring so the demo has zero extra model downloads by default —
    swap in the commented-out CrossEncoder block for the real thing.
    """
    query_terms = set(query.lower().split())

    def overlap_score(text):
        text_terms = set(text.lower().split())
        return len(query_terms & text_terms)

    scored = sorted(candidates, key=lambda c: overlap_score(c[1]), reverse=True)
    return scored[:top_n]

    # --- Real cross-encoder version (uncomment to use) ---
    # from sentence_transformers import CrossEncoder
    # model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # pairs = [(query, text) for _, text in candidates]
    # scores = model.predict(pairs)
    # ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    # return [c for c, s in ranked[:top_n]]


def answer_with_context(query: str, context_chunks: list):
    """Send retrieved context + query to the local LLM (grounded generation)."""
    context_text = "\n\n".join(f"[{cid}] {text}" for cid, text in context_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context doesn't contain the answer, say so explicitly.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]


def main():
    print("=== Module 1: RAG Pipeline Demo ===\n")

    chunk_records = build_corpus()
    print(f"Ingested {len(DOCUMENTS)} documents -> {len(chunk_records)} chunks\n")

    print("Embedding chunks and storing in ChromaDB...")
    collection = embed_and_store(chunk_records)

    query = "Why does my QuickETL job time out on a Datalake bucket?"
    print(f"\nQuery: {query}\n")

    candidates = hybrid_search(collection, chunk_records, query, top_k=5)
    print(f"Hybrid search returned {len(candidates)} candidate chunks:")
    for cid, text in candidates:
        print(f"  - {cid}: {text[:80]}...")

    top_chunks = rerank(query, candidates, top_n=2)
    print(f"\nAfter re-ranking, top {len(top_chunks)} chunks sent to LLM:")
    for cid, text in top_chunks:
        print(f"  - {cid}")

    print("\nGenerating grounded answer with local LLM...\n")
    answer = answer_with_context(query, top_chunks)
    print(f"Answer:\n{answer}")


if __name__ == "__main__":
    main()
