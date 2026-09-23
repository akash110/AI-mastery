"""
Capstone: End-to-End Oncall Assistant
=======================================
Ties all 4 modules together into one realistic mini-project — the kind of
system a data engineering team could actually build to reduce oncall load.

Flow for an incoming alert/question:
  1. [Module 4] Moderate the input.
  2. [Module 1] RAG: retrieve relevant runbook chunks (hybrid search + rerank).
  3. [Module 4] RBAC-filter retrieval so oncall-tier matters (e.g. a
     level-1 responder can't pull a "senior-only" runbook with prod creds).
  4. [Module 2] Agent: reason over the runbook + call a diagnostic tool
     (check_service_health) before proposing/executing a fix, gated by a
     deterministic guardrail.
  5. [Module 3] Evaluate the drafted incident summary for groundedness
     before it's allowed to page/notify anyone.
  6. [Module 4] Redact any PII from the final summary before it's logged.

This is intentionally a single file so you can read it top-to-bottom as a
narrative. In production you'd split these into services, but the *logic*
is the same regardless of scale.
"""

import json
import re
import ollama
import chromadb

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
MAX_AGENT_ITERATIONS = 5
GROUNDEDNESS_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Knowledge base: runbooks tagged with required oncall tier (RBAC)
# ---------------------------------------------------------------------------

RUNBOOKS = [
    {
        "id": "rb_timeout",
        "text": (
            "QuickETL Pipeline Timeout (ERR_504_TIMEOUT): caused by FGAC-enabled "
            "Datalake buckets missing the assumeRole parameter on SparkLoaderStep. "
            "Fix: add assumeRole=datalake-reader-role to the operator config."
        ),
        "required_tier": "any",
    },
    {
        "id": "rb_prod_creds",
        "text": (
            "Rotating prod database credentials: requires senior-oncall approval. "
            "Run rotate_prod_creds.sh only after confirming with the on-call lead, "
            "as this invalidates all active connections for 30 seconds."
        ),
        "required_tier": "senior",
    },
]


def build_kb():
    client = chromadb.Client()
    collection = client.get_or_create_collection("oncall_kb")
    for rb in RUNBOOKS:
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=rb["text"])["embedding"]
        collection.add(
            ids=[rb["id"]],
            embeddings=[embedding],
            documents=[rb["text"]],
            metadatas=[{"required_tier": rb["required_tier"]}],
        )
    return collection


TIER_ACCESS = {"l1_responder": {"any"}, "senior_oncall": {"any", "senior"}}


def retrieve_runbook(collection, query: str, oncall_tier: str):
    """Module 1 (retrieval) + Module 4 (RBAC filter) combined."""
    allowed = list(TIER_ACCESS.get(oncall_tier, {"any"}))
    embedding = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        where={"required_tier": {"$in": allowed}},
    )
    if not results["ids"][0]:
        return None
    return results["documents"][0][0]


# ---------------------------------------------------------------------------
# Diagnostic tool + deterministic guardrail (Module 2)
# ---------------------------------------------------------------------------

_SERVICE_STATE = {"quicketl-ingest-worker": "unhealthy"}


def check_service_health(service_name: str) -> str:
    return json.dumps({"service": service_name, "status": _SERVICE_STATE.get(service_name, "unknown")})


# ---------------------------------------------------------------------------
# Module 4: input moderation + PII redaction (reused from module4)
# ---------------------------------------------------------------------------

BLOCKED_KEYWORDS = ["ignore previous instructions", "bypass", "jailbreak"]
PII_PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def moderate_input(prompt: str):
    lowered = prompt.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in lowered:
            return False, kw
    return True, None


def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label}]", text)
    return text


# ---------------------------------------------------------------------------
# Module 3: groundedness check before the summary is allowed to page anyone
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict evaluator. Given CONTEXT and an INCIDENT SUMMARY, \
score groundedness 1-5: does every factual claim in the SUMMARY trace back to the CONTEXT?
CONTEXT: {context}
INCIDENT SUMMARY: {summary}
Respond ONLY with JSON: {{"groundedness": <int>, "reasoning": "<one sentence>"}}
"""


def check_groundedness(context: str, summary: str) -> dict:
    prompt = JUDGE_PROMPT.format(context=context, summary=summary)
    response = ollama.generate(model=LLM_MODEL, prompt=prompt, format="json")
    try:
        return json.loads(response["response"])
    except json.JSONDecodeError:
        return {"groundedness": 0, "reasoning": "judge parse failure"}


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------

def handle_oncall_query(user_query: str, oncall_tier: str, kb_collection):
    print(f"\n{'='*70}\nIncoming query (tier={oncall_tier}): {user_query}\n{'='*70}")

    # Step 1 [Module 4]: moderate
    allowed, blocked_kw = moderate_input(user_query)
    if not allowed:
        print(f"BLOCKED at moderation gate (matched: '{blocked_kw}')")
        return

    # Step 2+3 [Module 1 + 4]: RBAC-filtered retrieval
    runbook_text = retrieve_runbook(kb_collection, user_query, oncall_tier)
    if runbook_text is None:
        print("No accessible runbook found for this tier/query.")
        return
    print(f"\nRetrieved runbook (tier-filtered):\n  {runbook_text}")

    # Step 4 [Module 2]: diagnostic tool call with a guardrail
    # (Guardrail here: only report the fix if health check confirms unhealthy.)
    service_name = "quicketl-ingest-worker"
    health = json.loads(check_service_health(service_name))
    print(f"\nDiagnostic check: {health}")

    if health["status"] != "unhealthy":
        print("Guardrail: service is healthy, no incident summary needed. Stopping.")
        return

    # Draft the incident summary grounded in the runbook + diagnostic result
    draft_prompt = (
        f"Using ONLY this runbook context, write a 2-sentence incident summary "
        f"for the on-call channel about {service_name} being unhealthy.\n\n"
        f"Runbook: {runbook_text}"
    )
    summary = ollama.generate(model=LLM_MODEL, prompt=draft_prompt)["response"]
    print(f"\nDrafted incident summary:\n  {summary}")

    # Step 5 [Module 3]: groundedness gate before paging anyone
    eval_result = check_groundedness(runbook_text, summary)
    print(f"\nGroundedness check: {eval_result}")

    if eval_result.get("groundedness", 0) < GROUNDEDNESS_THRESHOLD:
        print("BLOCKED: summary failed groundedness threshold, escalating to human instead of auto-posting.")
        return

    # Step 6 [Module 4]: redact PII before logging/posting
    final_summary = redact_pii(summary)
    print(f"\nFINAL (redacted) summary posted to oncall channel:\n  {final_summary}")


def main():
    print("=== Capstone: Oncall Assistant (RAG + Agent + Eval + Governance) ===")
    kb_collection = build_kb()

    handle_oncall_query(
        "The quicketl-ingest-worker keeps failing with a timeout, what's going on?",
        oncall_tier="l1_responder",
        kb_collection=kb_collection,
    )

    handle_oncall_query(
        "How do I rotate prod database credentials right now?",
        oncall_tier="l1_responder",  # should be blocked by RBAC — wrong tier
        kb_collection=kb_collection,
    )


if __name__ == "__main__":
    main()
